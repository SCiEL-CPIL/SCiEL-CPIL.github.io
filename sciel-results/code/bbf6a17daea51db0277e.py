"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 26

# EVOLVE-BLOCK-START

def _objective(params):
    """Negative sum of radii (to minimize). params: [x1..xN, y1..yN, r1..rN]"""
    n = len(params) // 3
    x = params[0:n]
    y = params[n:2*n]
    r = params[2*n:3*n]
    
    # Constraint violations
    # Wall: x >= r, x <= 1-r, y >= r, y <= 1-r
    wall_viol = np.maximum(0, r - x) + np.maximum(0, x - (1 - r)) + \
                np.maximum(0, r - y) + np.maximum(0, y - (1 - r))
    
    # Pairwise: dist >= ri + rj
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = x[i_idx] - x[j_idx]
    dy = y[i_idx] - y[j_idx]
    dist = np.sqrt(dx**2 + dy**2 + 1e-12)
    pair_viol = np.maximum(0, (r[i_idx] + r[j_idx]) - dist)
    
    total_viol = np.sum(wall_viol) + np.sum(pair_viol)
    
    # Objective: maximize sum(r) -> minimize -sum(r) + penalty
    penalty_weight = 100.0
    return -np.sum(r) + penalty_weight * total_viol


def _is_valid(circles, tol=1e-6):
    """Check if circles are valid: inside [0,1]^2 and disjoint interiors."""
    n = len(circles)
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    # All radii positive
    if np.any(r < -tol):
        return False
    
    # Wall constraints
    if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
        return False
    
    # Pairwise constraints
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = x[i_idx] - x[j_idx]
    dy = y[i_idx] - y[j_idx]
    dist = np.sqrt(dx**2 + dy**2)
    if np.any(dist < r[i_idx] + r[j_idx] - tol):
        return False
    
    return True


def _post_process(circles):
    """Fix small numerical violations by shrinking radii slightly if needed."""
    circles = circles.copy()
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    # Clamp to unit square
    r = np.maximum(r, 1e-9)
    x = np.clip(x, r, 1 - r)
    y = np.clip(y, r, 1 - r)
    circles[:, 0] = x
    circles[:, 1] = y
    circles[:, 2] = r
    
    return circles


def _make_grid_initial():
    """Create a grid-based initial guess for 26 circles."""
    # Use a 5x5 grid, take first 25, add 1 more
    side = 5
    cells = []
    for i in range(side):
        for j in range(side):
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            # Radius limited by distance to walls and neighbors
            r = 0.5 / side
            cells.append([cx, cy, r])
    
    # We need 26, add one more in a reasonable position
    # Place at center with small radius
    cells.append([0.5, 0.5, 0.05])
    
    return np.array(cells[:N])


def _local_optimize(circles, time_limit=None):
    """Run local optimization starting from given circles."""
    n = len(circles)
    x0 = np.concatenate([circles[:, 0], circles[:, 1], circles[:, 2]])
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((1e-6, 1 - 1e-6))  # x
    for i in range(n):
        bounds.append((1e-6, 1 - 1e-6))  # y
    for i in range(n):
        bounds.append((1e-6, 0.5))  # r
    
    result = minimize(
        _objective,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    
    if result.success or result.fun < _objective(x0):
        x = result.x[0:n]
        y = result.x[n:2*n]
        r = result.x[2*n:3*n]
        return np.column_stack([x, y, r])
    
    return circles


def _differential_evolution_search(time_budget):
    """Run differential evolution for global search."""
    n = N
    # Bounds for DE
    bounds = []
    for i in range(n):
        bounds.append((0.01, 0.99))  # x
    for i in range(n):
        bounds.append((0.01, 0.99))  # y
    for i in range(n):
        bounds.append((0.01, 0.4))  # r
    
    # Use a reasonable number of iterations based on time budget
    # DE with 78 variables is expensive, so use modest popsize and maxiter
    popsize = 20
    maxiter = 30
    
    start_time = time.time()
    
    def _de_obj(params):
        # Early termination check
        if time.time() - start_time > time_budget * 0.8:
            return 1e6  # Large penalty to discourage further exploration
        
        return _objective(params)
    
    try:
        de_result = differential_evolution(
            _de_obj,
            bounds,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-6,
            seed=42,
            init='sobol',
            polish=False,  # We'll do our own polish
            update_interval=1,
            callback=None
        )
        
        if de_result.fun < 1e5:  # Reasonable result
            x = de_result.x[0:n]
            y = de_result.x[n:2*n]
            r = de_result.x[2*n:3*n]
            return np.column_stack([x, y, r])
    except Exception:
        pass
    
    return None


def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    best_circles = None
    best_score = -np.inf
    
    def _score(circles):
        if circles is None:
            return -np.inf
        if not _is_valid(circles, tol=1e-5):
            # Try to fix
            circles = _post_process(circles)
            if not _is_valid(circles, tol=1e-4):
                return -np.inf
        return np.sum(circles[:, 2])
    
    # Strategy 1: Grid initial + local optimization
    grid_init = _make_grid_initial()
    grid_opt = _local_optimize(grid_init)
    grid_score = _score(grid_opt)
    if grid_score > best_score:
        best_score = grid_score
        best_circles = grid_opt
    
    if time.time() - start_time > deadline:
        return _post_process(best_circles)
    
    # Strategy 2: Differential Evolution for global search
    de_time_budget = min(60.0, deadline - (time.time() - start_time))
    if de_time_budget > 10:
        de_result = _differential_evolution_search(de_time_budget)
        if de_result is not None:
            de_opt = _local_optimize(de_result)
            de_score = _score(de_opt)
            if de_score > best_score:
                best_score = de_score
                best_circles = de_opt
    
    if time.time() - start_time > deadline:
        return _post_process(best_circles)
    
    # Strategy 3: Multiple random restarts with local optimization
    rng = np.random.RandomState(42)
    remaining_time = deadline - (time.time() - start_time)
    max_restarts = int(remaining_time / 5.0)  # Estimate ~5s per restart
    max_restarts = max(1, min(max_restarts, 20))
    
    for restart in range(max_restarts):
        if time.time() - start_time > deadline:
            break
        
        # Random initialization
        x = rng.uniform(0.05, 0.95, N)
        y = rng.uniform(0.05, 0.95, N)
        r = rng.uniform(0.02, 0.15, N)
        init_circles = np.column_stack([x, y, r])
        
        # Local optimization
        opt_circles = _local_optimize(init_circles)
        opt_score = _score(opt_circles)
        
        if opt_score > best_score:
            best_score = opt_score
            best_circles = opt_circles
    
    # Final polish: high-precision local optimization on best solution
    if best_circles is not None and time.time() - start_time < deadline - 5:
        final_opt = _local_optimize(best_circles)
        final_score = _score(final_opt)
        if final_score > best_score:
            best_circles = final_opt
            best_score = final_score
    
    # Ensure validity and return
    if best_circles is None:
        best_circles = _make_grid_initial()
    
    best_circles = _post_process(best_circles)
    
    # Final validation
    if not _is_valid(best_circles, tol=1e-4):
        # Fallback to grid
        fallback = _make_grid_initial()
        return _post_process(fallback)
    
    return best_circles

# EVOLVE-BLOCK-END
