import math
import time
import numpy as np

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 8.0
EPS = 1e-12

# 6 unit normals of the (unrotated) hexagon
HEX_NORMALS = np.array([
    [math.cos(math.pi / 6.0 + k * math.pi / 3.0),
     math.sin(math.pi / 6.0 + k * math.pi / 3.0)]
    for k in range(6)
])

def _hex_normals(beta):
    c = math.cos(beta)
    s = math.sin(beta)
    R = np.array([[c, -s], [s, c]])
    return HEX_NORMALS @ R.T

def _enclosing_side(pts, cx, cy, beta):
    """Exact side length of smallest enclosing hexagon with rotation beta centered at (cx,cy)."""
    normals = _hex_normals(beta)
    centered = pts - np.array([cx, cy])
    projs = centered @ normals.T  # (N,6)
    D = max(np.max(projs), -np.min(projs))
    if D < 0:
        D = 0.0
    return (D + APOTHEM) / APOTHEM, D

def _overlap_violation(pts):
    """Sum of squared penetration depths for all pairs (Minkowski hexagon side 2, apothem sqrt(3))."""
    n = len(pts)
    viol = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = pts[i] - pts[j]
            projs = np.abs(d @ HEX_NORMALS.T)
            min_proj = np.min(projs)
            if min_proj < math.sqrt(3):
                v = math.sqrt(3) - min_proj
                viol += v * v
    return viol

def _objective(params, beta_mod=True):
    n_pts = N
    pts = params[:2 * n_pts].reshape(n_pts, 2)
    cx = params[2 * n_pts]
    cy = params[2 * n_pts + 1]
    beta = params[2 * n_pts + 2]
    if beta_mod:
        beta = ((beta % (math.pi / 6.0)) + math.pi / 6.0) % (math.pi / 6.0)
    S, _ = _enclosing_side(pts, cx, cy, beta)
    viol = _overlap_violation(pts)
    # Heavy penalty for overlap
    return S + 1000.0 * viol

def _random_init(scale=4.0):
    # Random cloud of N points
    rng = np.random.default_rng()
    pts = rng.normal(0.0, scale, size=(N, 2))
    cx = float(np.mean(pts[:, 0]))
    cy = float(np.mean(pts[:, 1]))
    beta = rng.uniform(0.0, math.pi / 6.0)
    params = np.concatenate([pts.ravel(), [cx, cy, beta]])
    return params

def _lattice_init(axial):
    pts = []
    for q, r in axial:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        pts.append((x, y))
    pts = np.array(pts)
    cx = float(np.mean(pts[:, 0]))
    cy = float(np.mean(pts[:, 1]))
    beta = 0.0
    params = np.concatenate([pts.ravel(), [cx, cy, beta]])
    return params

def _anneal(params0, t0, t1, iters, rng):
    """Simple simulated annealing with Gaussian perturbations."""
    best_params = params0.copy()
    best_val = _objective(params0)
    cur_params = params0.copy()
    cur_val = best_val
    n_params = len(params0)
    for it in range(iters):
        t = t0 + (t1 - t0) * (it / max(1, iters - 1))
        # Scale of perturbation
        sigma = 0.5 * t
        new_params = cur_params + rng.normal(0.0, sigma, size=n_params)
        new_val = _objective(new_params)
        if new_val < cur_val or rng.random() < math.exp(-(new_val - cur_val) / max(t, 1e-12)):
            cur_params = new_params
            cur_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_params = new_params.copy()
        # Periodic time check
        if it % 200 == 0 and time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
    return best_params, best_val

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # A few diverse lattice seeds (compact shapes)
    lattice_seeds = [
        [(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(5,1)],
        [(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)],
        [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3)],
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)],
        [(0,0),(1,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(0,4),(1,4)],
    ]

    best_params = None
    best_S = float('inf')

    rng = np.random.default_rng(42)

    # Phase 1: Try lattice seeds with annealing
    for seed in lattice_seeds:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        p0 = _lattice_init(seed)
        # Quick exact eval
        S0, _ = _enclosing_side(p0[:22].reshape(11,2), p0[22], p0[23], p0[24])
        if S0 < best_S:
            best_S = S0
            best_params = p0.copy()
        # Anneal
        p_best, v_best = _anneal(p0, t0=1.0, t1=0.01, iters=800, rng=rng)
        if v_best < best_S:
            best_S = v_best
            best_params = p_best.copy()

    # Phase 2: Random starts
    n_random = 30
    for i in range(n_random):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        p0 = _random_init(scale=3.0 + 2.0 * rng.random())
        p_best, v_best = _anneal(p0, t0=2.0, t1=0.005, iters=1200, rng=rng)
        if v_best < best_S:
            best_S = v_best
            best_params = p_best.copy()

    # Phase 3: Local polish with Nelder-Mead on the best
    if best_params is not None and time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        from scipy.optimize import minimize
        try:
            res = minimize(lambda p: _objective(p), best_params, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    if best_params is None:
        # Fallback: simple grid
        pts = np.array([[i, 0] for i in range(N)])
        best_params = np.concatenate([pts.ravel(), [0.0, 0.0, 0.0]])
        best_S, _ = _enclosing_side(pts, 0.0, 0.0, 0.0)

    # Extract
    pts = best_params[:22].reshape(11, 2)
    cx = float(best_params[22])
    cy = float(best_params[23])
    beta = float(best_params[24])
    beta = ((beta % (math.pi / 6.0)) + math.pi / 6.0) % (math.pi / 6.0)

    # Final exact S
    S_final, D = _enclosing_side(pts, cx, cy, beta)

    # Slight scale-up to guarantee disjointness (if any residual overlap due to numeric)
    # Compute overlap; if any, scale points about center slightly
    viol = _overlap_violation(pts)
    if viol > 1e-12:
        # Scale up until no overlap (max 1.01)
        for scale in np.linspace(1.0001, 1.01, 100):
            pts_s = cx + (pts - cx) * scale
            if _overlap_violation(pts_s) < 1e-12:
                pts = pts_s
                break
        S_final, D = _enclosing_side(pts, cx, cy, beta)

    angle_deg = math.degrees(beta)

    inner = [[float(x), float(y), float(angle_deg)] for x, y in pts]
    center = [float(cx), float(cy)]
    side = float(S_final)
    rotation = float(angle_deg)

    return inner, center, side, rotation
