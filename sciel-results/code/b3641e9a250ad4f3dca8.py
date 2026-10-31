"""Circle packing in unit square - 26 circles, maximize sum of radii."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

_DEADLINE = 140.0  # safety margin before 150s kill


def _total_objective(params, n, lam=500.0):
    """Combined objective: negative sum of radii + penalty for violations."""
    pts = params.reshape(n, 3)
    x, y, r = pts[:, 0], pts[:, 1], pts[:, 2]
    
    # Main objective: maximize sum of radii
    obj = -np.sum(r)
    
    # Penalty terms
    penalty = 0.0
    # Boundary constraints: circle must be inside [0,1]^2
    penalty += np.sum(np.maximum(0, r - x)) ** 2
    penalty += np.sum(np.maximum(0, r - (1 - x))) ** 2
    penalty += np.sum(np.maximum(0, r - y)) ** 2
    penalty += np.sum(np.maximum(0, r - (1 - y))) ** 2
    # Non-overlap: distance >= r_i + r_j
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
            min_d = r[i] + r[j]
            if d < min_d:
                penalty += (min_d - d) ** 2
    # Radii must be positive
    penalty += np.sum(np.maximum(0, -r)) ** 2
    
    return obj + lam * penalty


def _is_valid(pts, tol=1e-9):
    """Check if configuration is valid (all constraints satisfied)."""
    n = len(pts)
    x, y, r = pts[:, 0], pts[:, 1], pts[:, 2]
    if np.any(r <= 0):
        return False
    # Boundary
    if np.any(x - r < -tol) or np.any(1 - x - r < -tol) or np.any(y - r < -tol) or np.any(1 - y - r < -tol):
        return False
    # Non-overlap
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
            if d < r[i] + r[j] - tol:
                return False
    return True


def _make_grid_initial(n):
    side = int(np.ceil(np.sqrt(n)))
    pts = []
    for i in range(side):
        for j in range(side):
            if len(pts) >= n:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.95
            pts.append([x, y, r])
    return np.array(pts[:n])


def _make_hex_initial(n):
    # Hexagonal packing pattern
    pts = []
    for rows in range(4, 8):
        cols = int(np.ceil(n / rows))
        if cols < 3:
            continue
        dx = 1.0 / (cols + 0.5)
        dy = 1.0 / (rows + 0.5)
        r = min(dx, dy) * 0.48
        count = 0
        for row in range(rows):
            offset = 0.0 if row % 2 == 0 else dx / 2
            for col in range(cols):
                if count >= n:
                    break
                x = (col + 0.5 + offset) * dx
                y = (row + 0.5) * dy
                x = np.clip(x, r, 1 - r)
                y = np.clip(y, r, 1 - r)
                pts.append([x, y, r])
                count += 1
        if len(pts) >= n:
            break
    if len(pts) < n:
        return _make_grid_initial(n)
    return np.array(pts[:n])


def _make_random_initial(n, seed=42):
    rng = np.random.RandomState(seed)
    pts = []
    for _ in range(n):
        r = rng.uniform(0.03, 0.15)
        x = rng.uniform(r, 1 - r)
        y = rng.uniform(r, 1 - r)
        pts.append([x, y, r])
    return np.array(pts)


def _post_process(pts, n):
    """Enforce constraints by shrinking radii and adjusting positions."""
    pts = pts.copy()
    for _ in range(20):
        changed = False
        # Clip to boundaries
        for i in range(n):
            x, y, r = pts[i]
            r = min(r, x, 1 - x, y, 1 - y)
            if r < 0.001:
                r = 0.001
            x = np.clip(x, r, 1 - r)
            y = np.clip(y, r, 1 - r)
            if abs(pts[i, 0] - x) > 1e-10 or abs(pts[i, 1] - y) > 1e-10 or abs(pts[i, 2] - r) > 1e-10:
                changed = True
            pts[i] = [x, y, r]
        
        # Resolve overlaps by shrinking
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt((pts[i, 0] - pts[j, 0]) ** 2 + (pts[i, 1] - pts[j, 1]) ** 2)
                min_d = pts[i, 2] + pts[j, 2]
                if d < min_d - 1e-9:
                    shrink = (min_d - d) / 2 + 1e-7
                    pts[i, 2] = max(0.001, pts[i, 2] - shrink)
                    pts[j, 2] = max(0.001, pts[j, 2] - shrink)
                    changed = True
        if not changed:
            break
    return pts


def _optimize_single(pts0, n, time_limit=15.0):
    """Run multi-stage optimization on a single initial configuration."""
    params = pts0.flatten().copy()
    best_params = params.copy()
    best_obj = _total_objective(params, n)
    
    t0 = time.time()
    
    # Stage 1: L-BFGS-B with moderate penalty
    try:
        res = minimize(
            lambda p: _total_objective(p, n, lam=100.0),
            params,
            method='L-BFGS-B',
            options={'maxiter': 300, 'maxfun': 3000, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if res.fun < best_obj:
            best_obj = res.fun
            best_params = res.x.copy()
    except Exception:
        pass
    
    if time.time() - t0 > time_limit * 0.4:
        return best_params.reshape(n, 3)
    
    # Stage 2: Nelder-Mead refinement with higher penalty
    try:
        res2 = minimize(
            lambda p: _total_objective(p, n, lam=500.0),
            best_params,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
        )
        if res2.fun < best_obj:
            best_obj = res2.fun
            best_params = res2.x.copy()
    except Exception:
        pass
    
    if time.time() - t0 > time_limit * 0.7:
        return best_params.reshape(n, 3)
    
    # Stage 3: Final L-BFGS-B with high penalty for fine-tuning
    try:
        res3 = minimize(
            lambda p: _total_objective(p, n, lam=1000.0),
            best_params,
            method='L-BFGS-B',
            options={'maxiter': 200, 'maxfun': 2000, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        if res3.fun < best_obj:
            best_obj = res3.fun
            best_params = res3.x.copy()
    except Exception:
        pass
    
    return best_params.reshape(n, 3)


def construct():
    n = N
    t_start = time.time()
    
    # Generate diverse initial configurations
    inits = [
        _make_grid_initial(n),
        _make_hex_initial(n),
        _make_random_initial(n, seed=42),
        _make_random_initial(n, seed=123),
        _make_random_initial(n, seed=456),
        _make_random_initial(n, seed=789),
        _make_random_initial(n, seed=1024),
        _make_random_initial(n, seed=2048),
    ]
    
    best_pts = None
    best_score = -np.inf
    
    for idx, init in enumerate(inits):
        elapsed = time.time() - t_start
        if elapsed > _DEADLINE - 5:
            break
        
        remaining = _DEADLINE - elapsed - 5
        if remaining < 2:
            break
        
        # Allocate time based on remaining iterations
        time_limit = min(15.0, remaining / max(1, len(inits) - idx))
        
        try:
            opt_pts = _optimize_single(init, n, time_limit=time_limit)
            opt_pts = _post_process(opt_pts, n)
            
            if _is_valid(opt_pts):
                score = np.sum(opt_pts[:, 2])
                if score > best_score:
                    best_score = score
                    best_pts = opt_pts.copy()
        except Exception:
            continue
    
    # If no valid solution found, fall back to grid
    if best_pts is None or not _is_valid(best_pts):
        best_pts = _make_grid_initial(n)
        best_pts = _post_process(best_pts, n)
    
    return best_pts
# EVOLVE-BLOCK-END
