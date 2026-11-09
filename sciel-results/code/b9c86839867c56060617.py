import numpy as np
import time
from scipy.optimize import minimize


def _is_feasible(centers, sides, tol=1e-7):
    """Check if the configuration is feasible."""
    n = len(sides)
    # Boundary check
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if s <= 0:
            return False
        if not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(s)):
            return False
        if cx - s/2 < -tol or cx + s/2 > 1 + tol:
            return False
        if cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False
    # Overlap check
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j])/2 - tol
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            if dx < t and dy < t:
                return False
    return True


def _sum_sides(sides):
    """Objective: sum of side lengths."""
    return np.sum(sides)


def _flatten(centers, sides):
    """Flatten centers and sides into a single vector."""
    n = len(sides)
    return np.concatenate([centers.flatten(), sides])


def _unflatten(x):
    """Unflatten a vector back into centers and sides."""
    n = 10
    centers = x[:2*n].reshape(n, 2)
    sides = x[2*n:]
    return centers, sides


def _penalty(x, tol=1e-7, weight=1e4):
    """Penalty for constraint violations on flattened vector."""
    centers, sides = _unflatten(x)
    n = len(sides)
    pen = 0.0
    # Boundary penalties
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if s <= 0:
            pen += weight * (1.0 - s)
        if cx - s/2 < -tol:
            pen += weight * (-tol - (cx - s/2))
        if cx + s/2 > 1 + tol:
            pen += weight * ((cx + s/2) - (1 + tol))
        if cy - s/2 < -tol:
            pen += weight * (-tol - (cy - s/2))
        if cy + s/2 > 1 + tol:
            pen += weight * ((cy + s/2) - (1 + tol))
    # Overlap penalties
    for i in range(n):
        for j in range(i+1, n):
            t = (sides[i] + sides[j])/2 - tol
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            if abs(dx) < t and abs(dy) < t:
                pen += weight * ((t - abs(dx)) + (t - abs(dy)))
    return pen


def _objective(x, tol=1e-7, weight=1e4):
    """Combined objective: maximize sum of sides minus penalty (negated for minimization)."""
    centers, sides = _unflatten(x)
    return -np.sum(sides) + _penalty(x, tol, weight)


def _make_2big_8small():
    """Create initial config: 2 large squares in top row, 8 small in bottom 2 rows."""
    n = 10
    centers = np.zeros((n, 2))
    sides = np.zeros(n)
    
    # 2 big squares in top: each 0.5 x 0.5
    s_big = 0.5
    centers[0] = [0.25, 0.75]
    centers[1] = [0.75, 0.75]
    sides[0] = s_big
    sides[1] = s_big
    
    # 8 small squares in bottom: 4 per row, each 0.25 x 0.25
    s_small = 0.25
    for i in range(4):
        # Row 1 (y=0.375)
        centers[2+i] = [(i+0.5)*s_small, 0.375]
        sides[2+i] = s_small
        # Row 2 (y=0.125)
        centers[6+i] = [(i+0.5)*s_small, 0.125]
        sides[6+i] = s_small
    
    return centers, sides


def _make_3big_7small():
    """Create initial config: 3 large squares on top, 7 smaller below."""
    n = 10
    centers = np.zeros((n, 2))
    sides = np.zeros(n)
    
    # 3 big squares in top row
    s_big = 1.0 / 3
    for i in range(3):
        centers[i] = [(i + 0.5) * s_big, 1.0 - s_big/2]
        sides[i] = s_big
    
    # 7 small squares in bottom area
    # Try a 4+3 arrangement
    s_small = 0.2
    # Row 1: 4 squares
    for i in range(4):
        centers[3+i] = [(i + 0.5) * s_small, 0.6]
        sides[3+i] = s_small
    # Row 2: 3 squares
    for i in range(3):
        centers[7+i] = [(i + 0.5) * s_small + 0.1, 0.3]
        sides[7+i] = s_small
    
    return centers, sides


def _make_grid_fallback():
    """Fallback grid configuration."""
    n = 10
    k = int(np.ceil(np.sqrt(n)))
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    centers = np.stack([cx, cy], axis=1).astype(np.float64)
    sides = np.full(n, side, dtype=np.float64)
    return centers, sides


def _nelder_mead_optimize(x0, tol=1e-7, max_iter=5000, time_limit=30):
    """Run Nelder-Mead optimization on flattened vector."""
    start = time.time()
    
    def obj(x):
        if time.time() - start > time_limit:
            return 1e10  # Force termination
        return _objective(x, tol)
    
    try:
        result = minimize(obj, x0, method='Nelder-Mead',
                         options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-8})
        return result.x
    except Exception:
        return x0


def _lbfgsb_optimize(x0, tol=1e-7, max_iter=1000, time_limit=30):
    """Run L-BFGS-B optimization on flattened vector with bounds."""
    start = time.time()
    n = 10
    
    # Define bounds: centers in [0,1], sides in [0.01, 1]
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))  # cx
        bounds.append((0.0, 1.0))  # cy
        bounds.append((0.01, 1.0))  # side
    
    def obj(x):
        if time.time() - start > time_limit:
            return 1e10
        return _objective(x, tol)
    
    try:
        result = minimize(obj, x0, method='L-BFGS-B', bounds=bounds,
                         options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-6})
        return result.x
    except Exception:
        return x0


def _local_optimize(centers, sides, max_iter=5000, tol=1e-7):
    """Coordinate descent local optimization (kept as fallback)."""
    n = len(sides)
    best_centers = centers.copy()
    best_sides = sides.copy()
    best_obj = _sum_sides(sides) - _penalty(_flatten(centers, sides), tol)
    
    step_size = 0.01
    for iteration in range(max_iter):
        improved = False
        for i in range(n):
            for delta in [step_size, -step_size]:
                new_cx = centers[i, 0] + delta
                new_centers = centers.copy()
                new_centers[i, 0] = new_cx
                new_obj = _sum_sides(sides) - _penalty(_flatten(new_centers, sides), tol)
                if new_obj > best_obj and _is_feasible(new_centers, sides, tol):
                    best_centers = new_centers
                    best_sides = sides.copy()
                    best_obj = new_obj
                    centers = new_centers
                    improved = True
            for delta in [step_size, -step_size]:
                new_cy = centers[i, 1] + delta
                new_centers = centers.copy()
                new_centers[i, 1] = new_cy
                new_obj = _sum_sides(sides) - _penalty(_flatten(new_centers, sides), tol)
                if new_obj > best_obj and _is_feasible(new_centers, sides, tol):
                    best_centers = new_centers
                    best_sides = sides.copy()
                    best_obj = new_obj
                    centers = new_centers
                    improved = True
            for delta in [step_size, -step_size]:
                new_s = sides[i] + delta
                if new_s <= 0:
                    continue
                new_sides = sides.copy()
                new_sides[i] = new_s
                new_obj = _sum_sides(new_sides) - _penalty(_flatten(centers, new_sides), tol)
                if new_obj > best_obj and _is_feasible(centers, new_sides, tol):
                    best_centers = centers.copy()
                    best_sides = new_sides
                    best_obj = new_obj
                    sides = new_sides
                    improved = True
        
        if not improved:
            step_size *= 0.5
            if step_size < 1e-6:
                break
    
    return best_centers, best_sides


def _optimize_from_config(centers, sides, tol=1e-7, time_budget=60):
    """Try multiple optimization methods from a given configuration."""
    start = time.time()
    x0 = _flatten(centers, sides)
    
    # Try Nelder-Mead
    if time.time() - start < time_budget * 0.4:
        x_nm = _nelder_mead_optimize(x0, tol, max_iter=3000, time_limit=time_budget * 0.35)
        c_nm, s_nm = _unflatten(x_nm)
        if _is_feasible(c_nm, s_nm, tol):
            val_nm = np.sum(s_nm)
            val_orig = np.sum(sides)
            if val_nm > val_orig:
                centers, sides = c_nm, s_nm
                x0 = x_nm
    
    # Try L-BFGS-B
    if time.time() - start < time_budget * 0.8:
        x_lb = _lbfgsb_optimize(x0, tol, max_iter=500, time_limit=time_budget * 0.35)
        c_lb, s_lb = _unflatten(x_lb)
        if _is_feasible(c_lb, s_lb, tol):
            val_lb = np.sum(s_lb)
            val_cur = np.sum(sides)
            if val_lb > val_cur:
                centers, sides = c_lb, s_lb
    
    # Final local polish
    if time.time() - start < time_budget:
        centers, sides = _local_optimize(centers, sides, max_iter=2000, tol=tol)
    
    return centers, sides


def squares_in_square10() -> tuple:
    """
    Place 10 axis-aligned squares in the unit square to maximize the sum of side lengths.
    Uses multiple initial configurations with Nelder-Mead, L-BFGS-B, and coordinate descent.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s kill
    
    best_centers, best_sides = None, None
    best_value = 0.0
    
    # Try multiple initial configurations
    initial_configs = [
        _make_2big_8small(),
        _make_3big_7small(),
        _make_grid_fallback(),
    ]
    
    # Add some random perturbations of the best configs
    np.random.seed(42)
    for _ in range(5):
        c, s = _make_2big_8small()
        c += np.random.normal(0, 0.02, c.shape)
        s += np.random.normal(0, 0.01, s.shape)
        s = np.clip(s, 0.05, 1.0)
        initial_configs.append((c, s))
    
    for idx, (centers, sides) in enumerate(initial_configs):
        if time.time() > deadline - 30:
            break
        
        # Verify initial feasibility
        if not _is_feasible(centers, sides):
            continue
        
        current_val = np.sum(sides)
        
        # Allocate time budget based on remaining time
        remaining = deadline - time.time()
        time_budget = min(60, remaining * 0.3)
        
        if time_budget < 5:
            break
        
        # Optimize from this configuration
        opt_centers, opt_sides = _optimize_from_config(
            centers, sides, tol=1e-7, time_budget=time_budget
        )
        
        if _is_feasible(opt_centers, opt_sides):
            opt_val = np.sum(opt_sides)
            if opt_val > best_value:
                best_value = opt_val
                best_centers = opt_centers
                best_sides = opt_sides
    
    # If no feasible result found, use grid fallback
    if best_centers is None:
        best_centers, best_sides = _make_grid_fallback()
    
    # Final check
    if not _is_feasible(best_centers, best_sides):
        best_centers, best_sides = _make_grid_fallback()
    
    return best_centers, best_sides
