"""Suite task: pack 100 equal spheres in the unit cube [-1/2,1/2]^3.
Entry point construct() -> (N,3) array of centres.
Score = min( min_{i<j} |c_i-c_j|/2 , min_i min_k (1/2 - |c_ik|) ).
Design: repulsion-energy local optimization (L-BFGS-B, analytical grads)
from multiple diverse initializations + a final high-precision polish,
all under an explicit wall-clock deadline guard."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = 140.0          # safety margin vs 150 s wall-clock kill
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
    pen = np.maximum(c - x, 0.0) + np.maximum(-c - x, 0.0)   # per coord
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

def _init_bcc():
    """Body-Centered Cubic lattice initialization."""
    # BCC unit cell has 2 atoms. We need 100 atoms -> 50 cells.
    # Let's try a 5x5x4 grid of BCC cells? 5*5*4 = 100 cells -> 200 atoms. Too many.
    # We need exactly 100.
    # Let's use a 4x5x5 grid of simple cubic sites, and add BCC centers?
    # 4*5*5 = 100 simple cubic sites.
    # If we just use SC, it's less dense.
    # Let's try to pack 100 points in a BCC-like arrangement.
    # A 4x4x4 BCC grid has 4*4*4*2 = 128 points.
    # A 4x4x3 BCC grid has 4*4*3*2 = 96 points. Close.
    # Let's generate a 4x4x4 BCC grid (128 pts) and take the 100 closest to center?
    # Or just generate a 4x4x4 SC grid (64 pts) and add 36 BCC centers?
    
    # Simpler: Generate a dense random cloud and cluster? No.
    # Let's generate a 5x5x4 SC grid (100 pts). This is the same as _init_grid with m=5?
    # _init_grid uses m=ceil(100^(1/3)) = 5. 5x5x5=125. It takes first 100.
    
    # Let's try a 4x4x4 BCC grid, take 100 points.
    nx, ny, nz = 4, 4, 4
    pts = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Corners
                pts.append((i, j, k))
                # Center
                pts.append((i + 0.5, j + 0.5, k + 0.5))
    pts = np.array(pts)
    # Scale to fit in [-0.5, 0.5]
    # Range is 0 to 4 (corners) or 0.5 to 3.5 (centers).
    # Max coord is 4. Min is 0.
    # Center of mass is 1.5, 1.5, 1.5.
    # Normalize: (p - 1.5) / 2.0 * 0.9 (shrink slightly to avoid walls)
    pts = (pts - 1.5) / 2.0 * 0.9
    # We have 128 points. Take 100.
    # To make it symmetric, we should probably take the ones closest to the center?
    # Or just the first 100? The first 100 in this loop are biased.
    # Let's sort by distance to center and take 100?
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

    inits = [_init_grid(), _init_bcc(), _init_jittered(), _init_random()]
    
    # Phase 1: Global search with softer potential
    K, beta_search = 1.0, 20.0
    # Phase 2: Local polish with sharper potential
    beta_polish = 100.0

    for it, c in enumerate(inits):
        if time.time() - t0 > _DEADLINE - 20.0: # Reserve time for polish
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
    # Use a higher beta for sharper minimum resolution
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
