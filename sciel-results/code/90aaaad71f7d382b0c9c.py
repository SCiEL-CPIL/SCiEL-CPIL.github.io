import numpy as np
from scipy.optimize import minimize
import itertools

N = 26

def construct():
    """
    Packs 26 circles in a unit square maximizing the sum of radii.
    Uses improved penalty-based optimization with better initialization,
    adaptive penalties, and local refinement.
    """
    best_sum = -1
    best_result = None
    
    # Precompute bounds
    bounds = []
    for _ in range(N):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r
    
    def unpack(v):
        x = v[0::3]
        y = v[1::3]
        r = v[2::3]
        return x, y, r
    
    def is_valid(x, y, r, tol=1e-6):
        """Check if configuration is valid (no overlaps, within bounds)."""
        if np.any(r < -tol):
            return False
        if np.any(x < r - tol) or np.any(x > 1 - r + tol):
            return False
        if np.any(y < r - tol) or np.any(y > 1 - r + tol):
            return False
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist = np.sqrt(dx**2 + dy**2)
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        np.fill_diagonal(r_sum, 0)
        if np.any(dist < r_sum - tol):
            return False
        return True
    
    def make_cost(penalty_weight):
        def cost(v):
            x, y, r = unpack(v)
            
            # Objective: Maximize sum of radii -> Minimize negative sum
            obj = -np.sum(r)
            
            # Compute all pairwise distances
            dx = x[:, np.newaxis] - x[np.newaxis, :]
            dy = y[:, np.newaxis] - y[np.newaxis, :]
            dist = np.sqrt(dx**2 + dy**2)
            
            # Sum of radii for each pair
            r_sum = r[:, np.newaxis] + r[np.newaxis, :]
            
            # Overlap violation
            violations = np.maximum(0, r_sum - dist)
            np.fill_diagonal(violations, 0)
            
            penalty_overlap = np.sum(violations**2)
            
            # Boundary penalties
            pen_x_low = np.maximum(0, r - x)
            pen_x_high = np.maximum(0, x - (1 - r))
            pen_y_low = np.maximum(0, r - y)
            pen_y_high = np.maximum(0, y - (1 - r))
            
            penalty_boundary = np.sum(pen_x_low**2) + np.sum(pen_x_high**2) + \
                              np.sum(pen_y_low**2) + np.sum(pen_y_high**2)
            
            total_penalty = penalty_overlap + penalty_boundary
            
            return obj + penalty_weight * total_penalty
        
        return cost
    
    def grid_init(rows, cols, jitter=0.0):
        """Generate grid-based initialization."""
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= N:
                    break
                cx = (j + 0.5) / cols
                cy = (i + 0.5) / rows
                x[idx] = cx + np.random.uniform(-jitter, jitter)
                y[idx] = cy + np.random.uniform(-jitter, jitter)
                r[idx] = 0.03
                idx += 1
        
        # Fill remaining with random positions
        if idx < N:
            x[idx:] = np.random.uniform(0.1, 0.9, N - idx)
            y[idx:] = np.random.uniform(0.1, 0.9, N - idx)
            r[idx:] = 0.03
        
        return x, y, r
    
    def random_init():
        """Generate random initialization."""
        x = np.random.uniform(0.05, 0.95, N)
        y = np.random.uniform(0.05, 0.95, N)
        r = np.full(N, 0.03)
        return x, y, r
    
    # Different initialization strategies
    init_strategies = [
        ('grid_5x5', lambda: grid_init(5, 5, jitter=0.02)),
        ('grid_4x7', lambda: grid_init(4, 7, jitter=0.02)),
        ('grid_7x4', lambda: grid_init(7, 4, jitter=0.02)),
        ('random', random_init),
    ]
    
    num_trials = 15
    
    for trial in range(num_trials):
        # Select initialization strategy
        strat_name, strat_func = init_strategies[trial % len(init_strategies)]
        x, y, r = strat_func()
        
        # Initial vector
        v0 = np.zeros(N * 3)
        v0[0::3] = x
        v0[1::3] = y
        v0[2::3] = r
        
        # Use moderate penalty weight initially
        penalty_weight = 50.0
        
        try:
            cost_func = make_cost(penalty_weight)
            res = minimize(cost_func, v0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8})
            
            # Extract solution
            x_sol, y_sol, r_sol = unpack(res.x)
            
            # Try to refine with higher penalty weight if not perfectly valid
            if not is_valid(x_sol, y_sol, r_sol, tol=1e-5):
                penalty_weight = 200.0
                cost_func = make_cost(penalty_weight)
                res = minimize(cost_func, res.x, method='L-BFGS-B', bounds=bounds, 
                               options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
                x_sol, y_sol, r_sol = unpack(res.x)
            
            # Verify validity strictly
            if is_valid(x_sol, y_sol, r_sol, tol=1e-6):
                current_sum = np.sum(r_sol)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = np.column_stack((x_sol, y_sol, r_sol))
                    
        except Exception as e:
            pass
    
    # If we have a valid result, try to improve it further with local optimization
    if best_result is not None:
        x_best, y_best, r_best = unpack(best_result.ravel())
        
        # Try fine-tuning with very high penalty
        penalty_weight = 500.0
        cost_func = make_cost(penalty_weight)
        v_best = best_result.ravel()
        
        try:
            res = minimize(cost_func, v_best, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
            x_sol, y_sol, r_sol = unpack(res.x)
            
            if is_valid(x_sol, y_sol, r_sol, tol=1e-6):
                current_sum = np.sum(r_sol)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = np.column_stack((x_sol, y_sol, r_sol))
        except Exception as e:
            pass
    
    if best_result is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_result
