# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 620.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def _base_inside(u, v):
    """Vectorized base corridor check."""
    horiz = (u >= -1e-9) & (u <= 1.0 + 1e-9) & (v >= -3.0 - 1e-9) & (v <= 1.0 + 1e-9)
    vert = (u >= -3.0 - 1e-9) & (u <= 1.0 + 1e-9) & (v >= -1e-9) & (v <= 1.0 + 1e-9)
    return horiz | vert

def _build_grid(subsample=1):
    """Build sofa grid. subsample=1 for full, >1 for subsampled."""
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    if subsample > 1:
        xs = xs[::subsample]
        ys = ys[::subsample]
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    return X, Y

def _densify_path(p):
    """Densify path according to evaluator rules."""
    dense = [p[0]]
    for i in range(len(p) - 1):
        a1 = p[i, 2]
        a2 = p[i + 1, 2]
        m = max(1, int(np.ceil(abs(a2 - a1) - 1e-9)))
        if m > 1:
            ts = np.linspace(0.0, 1.0, m + 1)
            for t in ts[1:-1]:
                row = p[i] * (1.0 - t) + p[i + 1] * t
                dense.append(row)
        dense.append(p[i + 1])
    return np.array(dense)

def _eval_path_score(path, X, Y, early_stop=True):
    """
    Calculate score for a path.
    path: (n, 3) array.
    Returns score (number of points * 0.0001).
    """
    # Add evaluator endpoints
    p = np.vstack([np.array([[0.0, 0.0, 0.0]]), path, np.array([[-2.0, 0.0, 90.0]])])
    
    # Densify path
    dense = _densify_path(p)
    
    if len(dense) == 0:
        return 0.0
        
    inside = np.ones(X.shape, dtype=bool)
    
    for (cx, cy, a_deg) in dense:
        a = np.deg2rad(a_deg)
        c, s = np.cos(a), np.sin(a)
        dx = X - cx
        dy = Y - cy
        u = c * dx + s * dy
        v = -s * dx + c * dy
        mask = _base_inside(u, v)
        inside &= mask
        if early_stop and not inside.any():
            break
            
    return np.sum(inside) * 0.0001

def _initial_paths():
    """Generate multiple initial path candidates."""
    paths = []
    
    # Path 1: Arc-like path
    paths.append(np.array([
        [-0.4, 0.4, 20.0],
        [-0.9, 0.5, 45.0],
        [-1.4, 0.3, 65.0],
        [-1.8, 0.1, 80.0],
    ], dtype=np.float64))
    
    # Path 2: More gradual
    paths.append(np.array([
        [-0.3, 0.3, 15.0],
        [-0.7, 0.4, 30.0],
        [-1.1, 0.4, 50.0],
        [-1.5, 0.2, 70.0],
        [-1.8, 0.0, 85.0],
    ], dtype=np.float64))
    
    # Path 3: Keep y higher longer
    paths.append(np.array([
        [-0.5, 0.6, 25.0],
        [-1.0, 0.6, 50.0],
        [-1.5, 0.4, 70.0],
        [-1.9, 0.1, 85.0],
    ], dtype=np.float64))
    
    # Path 4: S-shaped
    paths.append(np.array([
        [-0.4, 0.2, 18.0],
        [-0.8, 0.5, 35.0],
        [-1.2, 0.5, 55.0],
        [-1.6, 0.3, 72.0],
        [-1.9, 0.0, 88.0],
    ], dtype=np.float64))
    
    # Path 5: 6 points
    paths.append(np.array([
        [-0.3, 0.3, 12.0],
        [-0.6, 0.4, 25.0],
        [-0.9, 0.5, 40.0],
        [-1.2, 0.5, 55.0],
        [-1.5, 0.3, 70.0],
        [-1.8, 0.1, 85.0],
    ], dtype=np.float64))
    
    # Path 6: 7 points - more detailed
    paths.append(np.array([
        [-0.2, 0.2, 10.0],
        [-0.5, 0.4, 22.0],
        [-0.8, 0.5, 35.0],
        [-1.1, 0.5, 48.0],
        [-1.4, 0.4, 62.0],
        [-1.7, 0.2, 76.0],
        [-1.9, 0.0, 88.0],
    ], dtype=np.float64))
    
    # Path 7: 8 points - very detailed
    paths.append(np.array([
        [-0.15, 0.2, 8.0],
        [-0.4, 0.35, 18.0],
        [-0.65, 0.5, 28.0],
        [-0.9, 0.55, 40.0],
        [-1.15, 0.5, 52.0],
        [-1.4, 0.35, 65.0],
        [-1.65, 0.15, 78.0],
        [-1.9, 0.0, 88.0],
    ], dtype=np.float64))
    
    # Path 8: 9 points - most detailed
    paths.append(np.array([
        [-0.1, 0.15, 6.0],
        [-0.3, 0.3, 14.0],
        [-0.5, 0.45, 24.0],
        [-0.7, 0.55, 34.0],
        [-0.9, 0.55, 45.0],
        [-1.1, 0.45, 56.0],
        [-1.35, 0.3, 68.0],
        [-1.6, 0.1, 80.0],
        [-1.9, 0.0, 88.0],
    ], dtype=np.float64))
    
    return paths

def _nelder_mead(f, x0, max_iter=2000, step=0.1, tol=1e-5, deadline=_DEADLINE - 30):
    n = len(x0)
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0
    for i in range(n):
        pt = x0.copy()
        pt[i] += step
        simplex[i + 1] = pt
    vals = np.array([f(simplex[i]) for i in range(n + 1)])
    
    for it in range(max_iter):
        if _elapsed() > deadline:
            break
        order = np.argsort(vals)
        simplex = simplex[order]
        vals = vals[order]
        if abs(vals[0] - vals[-1]) < tol:
            break
            
        centroid = simplex[:-1].mean(axis=0)
        xr = centroid + (centroid - simplex[-1])
        fr = f(xr)
        
        if fr < vals[0]:
            xe = centroid + 2.0 * (centroid - simplex[-1])
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe
                vals[-1] = fe
            else:
                simplex[-1] = xr
                vals[-1] = fr
        elif fr < vals[-2]:
            simplex[-1] = xr
            vals[-1] = fr
        else:
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            fc = f(xc)
            if fc < vals[-1]:
                simplex[-1] = xc
                vals[-1] = fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    vals[i] = f(simplex[i])
                    
    return simplex[int(np.argmin(vals))], np.min(vals)

def _refine_path(path, X, Y, max_iter=300, step=0.03, deadline=_DEADLINE - 30):
    """Refine a path using Nelder-Mead."""
    x0 = path.flatten()
    
    def f(flat):
        p = flat.reshape(-1, 3)
        return -_eval_path_score(p, X, Y, early_stop=True)
    
    xopt, _ = _nelder_mead(f, x0, max_iter=max_iter, step=step, deadline=deadline)
    return xopt.reshape(-1, 3)

def _random_restart_refine(path, X, Y, n_restarts=5, max_iter=100, step=0.05, deadline=_DEADLINE - 30):
    """Try random restarts around the given path to escape local optima."""
    best_path = path.copy()
    best_score = _eval_path_score(path, X, Y, early_stop=True)
    
    for _ in range(n_restarts):
        if _elapsed() > deadline:
            break
        
        # Create a perturbed version
        noise = np.random.normal(0, step, size=path.shape)
        # Keep angles in reasonable range
        noise[:, 2] = np.clip(noise[:, 2], -30, 30)
        perturbed = path + noise
        
        refined = _refine_path(perturbed, X, Y, max_iter=max_iter, step=step*0.5, deadline=deadline)
        score = _eval_path_score(refined, X, Y, early_stop=True)
        
        if score > best_score:
            best_path = refined
            best_score = score
    
    return best_path, best_score

def moving_sofa_2d() -> np.ndarray:
    # Use subsampled grid for optimization
    X_opt, Y_opt = _build_grid(subsample=4)
    X_full, Y_full = _build_grid(subsample=1)
    
    # Start with best initial path
    init_paths = _initial_paths()
    best_path = init_paths[0]
    best_score = _eval_path_score(best_path, X_opt, Y_opt)
    
    # Evaluate all initial paths
    for path in init_paths[1:]:
        if _elapsed() > _DEADLINE - 120:
            break
        score = _eval_path_score(path, X_opt, Y_opt)
        if score > best_score:
            best_path = path
            best_score = score
    
    # Optimize each initial path
    for i, path in enumerate(init_paths):
        if _elapsed() > _DEADLINE - 60:
            break
        if len(path) != len(best_path):
            # Only optimize paths with same length as current best
            continue
        
        refined = _refine_path(path, X_opt, Y_opt, max_iter=400, step=0.05)
        score = _eval_path_score(refined, X_opt, Y_opt)
        if score > best_score:
            best_path = refined
            best_score = score
    
    # Try different path lengths
    for n_pts in [3, 4, 5, 6, 7, 8, 9]:
        if _elapsed() > _DEADLINE - 45:
            break
        
        # Create initial path with n_pts by interpolating best_path
        if len(best_path) != n_pts:
            ts = np.linspace(0, 1, n_pts)
            init_pts = np.zeros((n_pts, 3))
            for i in range(n_pts):
                t = ts[i] * (len(best_path) - 1)
                idx = int(t)
                frac = t - idx
                if idx >= len(best_path) - 1:
                    init_pts[i] = best_path[min(idx, len(best_path) - 1)]
                else:
                    init_pts[i] = best_path[idx] * (1 - frac) + best_path[idx + 1] * frac
        else:
            init_pts = best_path.copy()
        
        refined = _refine_path(init_pts, X_opt, Y_opt, max_iter=300, step=0.04)
        score = _eval_path_score(refined, X_opt, Y_opt)
        if score > best_score:
            best_path = refined
            best_score = score
    
    # Random restart refinement on subsampled grid
    if _elapsed() < _DEADLINE - 30:
        best_path, best_score = _random_restart_refine(
            best_path, X_opt, Y_opt, 
            n_restarts=3, max_iter=80, step=0.08, 
            deadline=_DEADLINE - 25
        )
    
    # Final refinement on full grid
    if _elapsed() < _DEADLINE - 20:
        refined_full = _refine_path(best_path, X_full, Y_full, max_iter=200, step=0.015, deadline=_DEADLINE - 10)
        score_full = _eval_path_score(refined_full, X_full, Y_full)
        score_best_full = _eval_path_score(best_path, X_full, Y_full)
        if score_full > score_best_full:
            best_path = refined_full
    
    # Second round of refinement if time permits
    if _elapsed() < _DEADLINE - 15:
        refined_full2 = _refine_path(best_path, X_full, Y_full, max_iter=100, step=0.008, deadline=_DEADLINE - 5)
        score_full2 = _eval_path_score(refined_full2, X_full, Y_full)
        score_best_full2 = _eval_path_score(best_path, X_full, Y_full)
        if score_full2 > score_best_full2:
            best_path = refined_full2
    
    # Random restart on full grid if time permits
    if _elapsed() < _DEADLINE - 10:
        best_path, _ = _random_restart_refine(
            best_path, X_full, Y_full,
            n_restarts=2, max_iter=50, step=0.03,
            deadline=_DEADLINE - 5
        )
    
    # Ensure valid output
    best_path = np.asarray(best_path, dtype=np.float64)
    if best_path.ndim != 2 or best_path.shape[1] != 3 or best_path.shape[0] < 1:
        best_path = init_paths[0]
    if not np.all(np.isfinite(best_path)):
        best_path = init_paths[0]
        
    return best_path
# EVOLVE-BLOCK-END
