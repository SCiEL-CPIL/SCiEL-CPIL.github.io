"""Suite task: pack 100 equal spheres in the unit cube [-1/2,1/2]^3.
Entry point construct() -> (N,3) array of centres.
Score = min( min_{i<j} |c_i-c_j|/2 , min_i min_k (1/2 - |c_ik|) ).
Design: hybrid lattice initialization (4x4x4 SC + interstitial FCC) +
repulsion-energy L-BFGS-B optimization with analytical gradients,
plus a final high-precision polish, all under an explicit wall-clock deadline guard."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = 145.0
_rng = np.random.default_rng(12345)


def _radius(c):
    """Common radius: min of half min pairwise distance and min wall clearance."""
    c = np.asarray(c, dtype=float)
    n = c.shape[0]
    d = 1e30
    for i in range(n - 1):
        diff = c[i + 1:] - c[i]
        d = min(d, float(np.min(np.einsum('ij,ij->i', diff, diff))))
    if d > 1e29:
        return float(np.min(0.5 - np.abs(c)))
    return min(0.5 * np.sqrt(d), float(np.min(0.5 - np.abs(c))))


def _energy(x, c, K, beta):
    """E = -K*min_pair_dist + beta*sum(wall_pen^2).  Grads analytic."""
    n = x.size
    c = np.asarray(c, dtype=float)
    # --- wall penalty (with gradient) ---
    pen = np.maximum(c - x, 0.0) + np.maximum(-c - x, 0.0)
    e_wall = beta * float(np.dot(pen, pen))
    g_wall = -2.0 * beta * (np.sign(x - c) * np.maximum(c - x, 0.0)
                            + np.sign(-c - x) * np.maximum(-c - x, 0.0))
    # --- repulsion (soft-min of pairwise distances) ---
    d = np.zeros(n)
    g = np.zeros(n)
    for i in range(n - 1):
        diff = x[i + 1:] - x[i]
        dist = np.sqrt(np.einsum('ij,ij->i', diff, diff))
        d[i + 1:] = dist
        g[i + 1:] = diff / dist
    dmin = float(np.min(d))
    m = -np.logaddexp.reduce(-beta * d)
    e_rep = -K * m
    w = np.exp(-beta * d - m)
    g = g * w[:, None]
    e_rep += K * float(np.sum(w))
    g_rep = K * g
    return e_rep + e_wall, g_rep + g_wall


def _init_grid():
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    g = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return g[:N].copy()


def _init_hybrid_lattice():
    """
    Hybrid initialization: 4x4x4 simple cubic lattice (64 points) +
    36 points in FCC-like interstitial positions.
    
    The 4x4x4 SC lattice has spacing s. We place additional points at
    face-center and body-center positions of the cubic cells to create
    a denser, more symmetric starting configuration.
    """
    # 4x4x4 SC lattice
    n_sc = 4
    s = 0.9 / (n_sc - 1)  # spacing to fit in [-0.45, 0.45]
    sc_pts = []
    for i in range(n_sc):
        for j in range(n_sc):
            for k in range(n_sc):
                x = -0.45 + i * s
                y = -0.45 + j * s
                z = -0.45 + k * s
                sc_pts.append([x, y, z])
    sc_pts = np.array(sc_pts)
    
    # Interstitial positions: face centers and body centers of the cubic cells
    # For a 4x4x4 grid, there are 3x3x3 = 27 internal cells
    # Each cell has 6 face centers and 1 body center
    # We'll use body centers (27) + some face centers (9 more) = 36
    inter_pts = []
    
    # Body centers of all 3x3x3 internal cells
    for i in range(n_sc - 1):
        for j in range(n_sc - 1):
            for k in range(n_sc - 1):
                x = -0.45 + (i + 0.5) * s
                y = -0.45 + (j + 0.5) * s
                z = -0.45 + (k + 0.5) * s
                inter_pts.append([x, y, z])
    
    # Add face centers of a subset of cells to reach 36
    # Use face centers on the x-faces of cells where i+j+k is even
    for i in range(n_sc - 1):
        for j in range(n_sc - 1):
            for k in range(n_sc - 1):
                if (i + j + k) % 2 == 0:
                    # Face center on the +x face of cell (i,j,k)
                    x = -0.45 + (i + 1) * s
                    y = -0.45 + (j + 0.5) * s
                    z = -0.45 + (k + 0.5) * s
                    inter_pts.append([x, y, z])
    
    # Take first 36 interstitial points
    inter_pts = np.array(inter_pts[:36])
    
    # Combine
    pts = np.vstack([sc_pts, inter_pts])
    return pts[:N]


def _init_fcc():
    """
    FCC lattice initialization.
    FCC has 4 atoms per unit cell. For 100 atoms, use a distorted FCC.
    A 5x5x4 FCC grid would have 5*5*4*4 = 400 atoms (too many).
    A 4x4x3 FCC grid would have 4*4*3*4 = 192 atoms.
    A 3x3x3 FCC grid would have 3*3*3*4 = 108 atoms. Close to 100.
    We'll use a 3x3x3 FCC grid (108 atoms) and take the 100 closest to center.
    """
    n = 3
    a = 0.9 / (n * np.sqrt(2))  # lattice parameter scaled to fit
    
    pts = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                # Corner
                pts.append([i * a, j * a, k * a])
                # Face centers
                pts.append([(i + 0.5) * a, (j + 0.5) * a, k * a])
                pts.append([(i + 0.5) * a, j * a, (k + 0.5) * a])
                pts.append([i * a, (j + 0.5) * a, (k + 0.5) * a])
    
    pts = np.array(pts)
    # Center the lattice
    center = pts.mean(axis=0)
    pts = pts - center
    # Scale to fit in [-0.45, 0.45]
    max_coord = np.abs(pts).max()
    if max_coord > 0:
        pts = pts * (0.45 / max_coord)
    
    # Take 100 closest to center
    dists = np.sum(pts**2, axis=1)
    idx = np.argsort(dists)[:N]
    return pts[idx]


def _init_random():
    return _rng.uniform(-0.49, 0.49, size=(N, 3))


def _init_jittered():
    return _init_grid() + _rng.normal(0, 0.012, size=(N, 3))


def construct():
    t0 = time.time()
    best = _init_grid()
    best_r = _radius(best)

    # Generate diverse initializations
    inits = [
        _init_grid(),
        _init_hybrid_lattice(),
        _init_fcc(),
        _init_jittered(),
        _init_random(),
    ]

    # Phase 1: Global search with softer potential
    K, beta_search = 1.0, 20.0
    # Phase 2: Local polish with sharper potential
    beta_polish = 100.0

    for it, c in enumerate(inits):
        if time.time() - t0 > _DEADLINE - 20.0:
            break
        c = np.clip(c, -0.49, 0.49)
        x0 = c.ravel()
        try:
            res = minimize(_energy, x0, args=(c, K, beta_search), method='L-BFGS-B',
                           jac=True, options={'maxiter': 200, 'maxls': 30,
                                              'gtol': 1e-6, 'ftol': 1e-10})
            cand = np.clip(res.x.reshape(N, 3), -0.5, 0.5)
            r = _radius(cand)
            if r > best_r + 1e-12:
                best, best_r = cand, r
        except Exception:
            pass

    # --- final high-precision polish from the best point ---
    if time.time() - t0 < _DEADLINE - 5.0:
        c = best.copy()
        for _ in range(2):
            if time.time() - t0 > _DEADLINE - 2.0:
                break
            try:
                res = minimize(_energy, c.ravel(), args=(c, K, beta_polish),
                               method='L-BFGS-B', jac=True,
                               options={'maxiter': 1000, 'maxls': 50,
                                        'gtol': 1e-12, 'ftol': 1e-15})
                cand = np.clip(res.x.reshape(N, 3), -0.5, 0.5)
                r = _radius(cand)
                if r > best_r + 1e-14:
                    best, best_r = cand, r
            except Exception:
                break

    # --- final validity guard: clamp strictly inside, keep finite ---
    best = np.clip(best, -0.4999999, 0.4999999)
    if not np.all(np.isfinite(best)):
        best = _init_grid()
    return best
