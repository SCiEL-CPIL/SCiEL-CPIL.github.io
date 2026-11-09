# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 620.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def _base_inside(u, v):
    """Vectorized base corridor check."""
    horiz = (u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)
    vert = (u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
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

def _eval_path_score(path, X, Y):
    """
    Calculate score for a path.
    path: (n, 3) array.
    Returns score (number of points * 0.0001).
    """
    # Add evaluator endpoints
    p = np.vstack([np.array([[0.0, 0.0, 0.0]]), path, np.array([[-2.0, 0.0, 90.0]])])
    
    # Densify path
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
    
    if not dense:
        return 0.0
        
    dense = np.array(dense)
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
        if not inside.any():
            break
            
    return np.sum(inside) * 0.0001

def _initial_path():
    # Better initial path: keep the inner corner of the L near the sofa's critical region
    # The sofa center is around (-0.75, 0). The L-corner should track near this.
    # At a=0, corner at (0,0); at a=90, corner at (-2,0)
    # Intermediate: move corner along an arc/curve that keeps sofa inside corridor
    return np.array([
        [-0.3, 0.3, 15.0],
        [-0.8, 0.4, 35.0],
        [-1.3, 0.3, 55.0],
        [-1.7, 0.1, 75.0],
    ], dtype=np.float64)

def _nelder_mead(f, x0, max_iter=2000, step=0.1, tol=1e-5):
    n = len(x0)
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0
    for i in range(n):
        pt = x0.copy()
        pt[i] += step
        simplex[i + 1] = pt
    vals = np.array([f(simplex[i]) for i in range(n + 1)])
    
    for it in range(max_iter):
        if _elapsed() > _DEADLINE - 30:
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

def _eval_path_fast(path, X, Y):
    """Faster evaluation: early exit and use float32 for speed."""
    p = np.vstack([np.array([[0.0, 0.0, 0.0]]), path, np.array([[-2.0, 0.0, 90.0]])])
    
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
    
    if not dense:
        return 0.0
        
    dense = np.array(dense, dtype=np.float32)
    inside = np.ones(X.shape, dtype=bool)
    Xf = X.astype(np.float32)
    Yf = Y.astype(np.float32)
    
    for (cx, cy, a_deg) in dense:
        a = np.deg2rad(a_deg)
        c, s = np.cos(a), np.sin(a)
        dx = Xf - cx
        dy = Yf - cy
        u = c * dx + s * dy
        v = -s * dx + c * dy
        horiz = (u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)
        vert = (u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
        mask = horiz | vert
        inside &= mask
        if not inside.any():
            return 0.0
            
    return np.sum(inside) * 0.0001

def moving_sofa_2d() -> np.ndarray:
    # Use subsampled grid for fast optimization
    X_opt, Y_opt = _build_grid(subsample=2)
    X_full, Y_full = _build_grid(subsample=1)
    
    best_path = _initial_path()
    best_score = _eval_path_fast(best_path, X_opt, Y_opt)
    
    # Try multiple starting points with different numbers of waypoints
    for n_pts in [3, 4, 5, 6]:
        if _elapsed() > _DEADLINE - 90:
            break
            
        # Create initial path with n_pts waypoints
        if len(best_path) >= n_pts:
            # Resample
            ts = np.linspace(0, 1, n_pts)
            init_pts = np.zeros((n_pts, 3))
            for i in range(n_pts):
                t = ts[i] * (len(best_path) - 1)
                idx = min(int(t), len(best_path) - 2)
                frac = t - idx
                init_pts[i] = best_path[idx] * (1 - frac) + best_path[idx+1] * frac
        else:
            # Pad by repeating last point
            init_pts = np.zeros((n_pts, 3))
            for i in range(n_pts):
                src_idx = min(int(i * len(best_path) / n_pts), len(best_path) - 1)
                init_pts[i] = best_path[src_idx]
        
        x0 = init_pts.flatten()
        
        def f(flat):
            path = flat.reshape(-1, 3)
            return -_eval_path_fast(path, X_opt, Y_opt)
            
        # Use Nelder-Mead with reasonable iterations
        xopt, _ = _nelder_mead(f, x0, max_iter=300, step=0.08)
        opt_path = xopt.reshape(-1, 3)
        
        opt_score = _eval_path_fast(opt_path, X_opt, Y_opt)
        if opt_score > best_score:
            best_path = opt_path
            best_score = opt_score
            
    # Refine on finer grid
    if _elapsed() < _DEADLINE - 60:
        X_ref, Y_ref = _build_grid(subsample=1)
        x0 = best_path.flatten()
        
        def f_ref(flat):
            path = flat.reshape(-1, 3)
            return -_eval_path_fast(path, X_ref, Y_ref)
        
        xopt, _ = _nelder_mead(f_ref, x0, max_iter=150, step=0.03)
        opt_path = xopt.reshape(-1, 3)
        opt_score = _eval_path_fast(opt_path, X_ref, Y_ref)
        if opt_score > best_score:
            best_path = opt_path
            best_score = opt_score
    
    # Final check on full grid
    if _elapsed() < _DEADLINE - 20:
        final_score = _eval_path_score(best_path, X_full, Y_full)
        if final_score > 0:
            # Try one more small refinement
            x0 = best_path.flatten()
            def f_full(flat):
                path = flat.reshape(-1, 3)
                return -_eval_path_score(path, X_full, Y_full)
            xopt, _ = _nelder_mead(f_full, x0, max_iter=50, step=0.01)
            opt_path = xopt.reshape(-1, 3)
            opt_score = _eval_path_score(opt_path, X_full, Y_full)
            if opt_score > final_score:
                best_path = opt_path

    # Ensure valid output
    best_path = np.asarray(best_path, dtype=np.float64)
    if best_path.ndim != 2 or best_path.shape[1] != 3 or best_path.shape[0] < 1:
        best_path = _initial_path()
    if not np.all(np.isfinite(best_path)):
        best_path = _initial_path()
        
    return best_path
# EVOLVE-BLOCK-END
