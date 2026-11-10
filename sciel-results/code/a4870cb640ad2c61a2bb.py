# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time
from scipy.optimize import minimize

def _compute_score(points):
    """Compute R = (d_min/d_max)^2 for a set of points."""
    d = pdist(points)
    d_min = np.min(d)
    d_max = np.max(d)
    if d_max == 0:
        return 0.0
    return (d_min / d_max) ** 2

def _build_symmetric(params):
    """
    Build 16 points with 4-fold rotational symmetry (90 degree rotations).
    params: array of shape (8,) representing 4 unique points in the first quadrant
            (and on the positive axes), each duplicated by 90-degree rotations.
    Each of the 4 base points p_k = (x_k, y_k) generates 4 points:
        (x_k, y_k), (-y_k, x_k), (-x_k, -y_k), (y_k, -x_k)
    This gives 16 points. We require the 4 base points to be distinct and
    in the first quadrant region to avoid duplicates.
    """
    pts = []
    for k in range(4):
        x = params[2*k]
        y = params[2*k+1]
        pts.append([x, y])
        pts.append([-y, x])
        pts.append([-x, -y])
        pts.append([y, -x])
    return np.array(pts, dtype=float)

def _neg_score_sym(params):
    pts = _build_symmetric(params)
    return -_compute_score(pts)

def _optimize_symmetric(params0, time_budget, rng):
    """Optimize the 8 symmetric parameters via Nelder-Mead + local search."""
    start = time.time()
    best_params = params0.copy()
    best_val = _neg_score_sym(params0)
    
    # Nelder-Mead on the 8-dim symmetric parameterization
    try:
        res = minimize(_neg_score_sym, params0, method='Nelder-Mead',
                       options={'maxiter': 4000, 'xatol': 1e-8, 'fatol': 1e-12,
                                'adaptive': True})
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x
    except Exception:
        pass
    
    if time.time() - start > time_budget:
        return best_params, -best_val
    
    # Multi-start local refinement around best
    for restart in range(3):
        if time.time() - start > time_budget:
            break
        pert = rng.normal(0, 0.02, size=8)
        p = best_params + pert
        try:
            res = minimize(_neg_score_sym, p, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-12,
                                    'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
    return best_params, -best_val

def _full_free_optimize(points0, time_budget, rng):
    """Free-form Nelder-Mead on all 32 coordinates as a fallback / refiner."""
    start = time.time()
    flat0 = points0.ravel().copy()
    best_flat = flat0.copy()
    best_val = -_compute_score(points0)
    
    def _neg(flat):
        pts = flat.reshape(-1, 2)
        return -_compute_score(pts)
    
    try:
        res = minimize(_neg, flat0, method='Nelder-Mead',
                       options={'maxiter': 6000, 'xatol': 1e-7, 'fatol': 1e-11,
                                'adaptive': True})
        if res.fun < best_val:
            best_val = res.fun
            best_flat = res.x
    except Exception:
        pass
    
    if time.time() - start > time_budget:
        return best_flat.reshape(-1, 2), -best_val
    
    # A few local restarts
    for _ in range(2):
        if time.time() - start > time_budget:
            break
        p = best_flat + rng.normal(0, 0.01, size=best_flat.size)
        try:
            res = minimize(_neg, p, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-7, 'fatol': 1e-11,
                                    'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_flat = res.x
        except Exception:
            pass
    return best_flat.reshape(-1, 2), -best_val

def _grid_search_symmetric(rng, time_budget):
    """Dense grid search over the 8 symmetric parameters."""
    start = time.time()
    best_params = None
    best_val = -np.inf
    
    # Reasonable ranges for first-quadrant base points
    # The 4 base points should roughly tile the first quadrant.
    # Use a coarse-to-fine grid.
    axes_ranges = [
        (0.0, 2.0, 11),   # x0
        (0.0, 2.0, 11),   # y0
        (0.0, 2.0, 9),    # x1
        (0.0, 2.0, 9),    # y1
        (0.0, 2.0, 9),    # x2
        (0.0, 2.0, 9),    # y2
        (0.0, 2.0, 9),    # x3
        (0.0, 2.0, 9),    # y3
    ]
    
    # Full cartesian product is too large (11*11*9^6 ~ 2.4M). 
    # Instead do a randomized dense sampling + local refine on good candidates.
    n_samples = 20000
    for _ in range(n_samples):
        if time.time() - start > time_budget:
            break
        params = np.empty(8)
        for k in range(8):
            lo, hi, _ = axes_ranges[k]
            params[k] = rng.uniform(lo, hi)
        # Encourage distinct base points by enforcing ordering in a loose sense
        val = -_neg_score_sym(params)
        if val > best_val:
            best_val = val
            best_params = params.copy()
    
    if best_params is None:
        return None, 0.0
    return best_params, best_val

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    start_time = time.time()
    deadline = start_time + 600  # safety margin before 630s kill
    
    rng = np.random.default_rng(12345)
    
    best_points = None
    best_score = 0.0
    
    def _update(pts, sc):
        nonlocal best_points, best_score
        if sc > best_score:
            best_score = sc
            best_points = pts.copy()
    
    # --- Phase 1: Symmetric parameterized search (main engine) ---
    # Good initial guesses for the 4 base points (first quadrant).
    # 4x4 grid in first quadrant: points at (0.5,0.5),(1.5,0.5),(0.5,1.5),(1.5,1.5)
    inits = [
        np.array([0.5, 0.5, 1.5, 0.5, 0.5, 1.5, 1.5, 1.5]),
        np.array([0.3, 0.3, 1.7, 0.3, 0.3, 1.7, 1.7, 1.7]),
        np.array([0.6, 0.4, 1.4, 0.6, 0.4, 1.4, 1.6, 1.6]),
        np.array([0.0, 0.7, 1.0, 0.0, 0.0, 1.0, 0.7, 1.0]),
        np.array([0.7, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.7]),
        np.array([0.5, 0.0, 1.0, 0.5, 0.5, 1.0, 0.0, 1.5]),
    ]
    
    # Grid/randomized dense search to find a good basin
    gp, gv = _grid_search_symmetric(rng, time_budget=60)
    if gp is not None:
        inits.append(gp)
        _update(_build_symmetric(gp), gv)
    
    # Optimize each initial guess with symmetric Nelder-Mead
    for init in inits:
        if time.time() > deadline - 60:
            break
        params, sc = _optimize_symmetric(init, time_budget=30, rng=rng)
        pts = _build_symmetric(params)
        sc = _compute_score(pts)
        _update(pts, sc)
    
    # --- Phase 2: Free-form refinement of the best symmetric result ---
    if best_points is not None:
        if time.time() < deadline - 30:
            fp, fs = _full_free_optimize(best_points, time_budget=40, rng=rng)
            _update(fp, fs)
    
    # --- Phase 3: Random restarts on symmetric parameterization ---
    while time.time() < deadline - 20:
        params = rng.uniform(0.0, 2.0, size=8)
        sc = _compute_score(_build_symmetric(params))
        if sc > best_score * 0.9:  # only refine promising ones
            p2, s2 = _optimize_symmetric(params, time_budget=15, rng=rng)
            pts = _build_symmetric(p2)
            _update(pts, _compute_score(pts))
        else:
            # quick evaluate
            pass
    
    # --- Phase 4: Final free-form polish ---
    if best_points is not None and time.time() < deadline - 10:
        fp, fs = _full_free_optimize(best_points, time_budget=20, rng=rng)
        _update(fp, fs)
    
    if best_points is None:
        # Fallback: 4x4 grid
        x = np.linspace(-1.5, 1.5, 4)
        y = np.linspace(-1.5, 1.5, 4)
        xx, yy = np.meshgrid(x, y)
        best_points = np.column_stack([xx.ravel(), yy.ravel()])
    
    # Ensure finite and correct shape
    if not np.all(np.isfinite(best_points)):
        x = np.linspace(-1.5, 1.5, 4)
        y = np.linspace(-1.5, 1.5, 4)
        xx, yy = np.meshgrid(x, y)
        best_points = np.column_stack([xx.ravel(), yy.ravel()])
    
    return best_points

# EVOLVE-BLOCK-END
