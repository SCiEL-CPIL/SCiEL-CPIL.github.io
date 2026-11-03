"""
Optimized circle packing in a unit square.
Uses L-BFGS-B with a smooth penalty function for non-overlap constraints.
Multi-start strategy to find better local optima.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 26
DEADLINE = 110.0  # Safety margin from 120s limit

def construct():
    start_time = time.time()
    
    # Initial best solution
    # Start with a grid configuration as a safe baseline
    side = int(np.ceil(np.sqrt(N)))
    grid_x = []
    grid_y = []
    grid_r = []
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= N:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 1e-4
            grid_x.append(x)
            grid_y.append(y)
            grid_r.append(r)
            idx += 1
            
    best_x = np.array(grid_x)
    best_y = np.array(grid_y)
    best_r = np.array(grid_r)
    best_val = np.sum(best_r)
    
    def check_validity(x, y, r):
        # Check boundary
        if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9):
            return False
        if np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
            return False
        if np.any(r < -1e-9):
            return False
        
        # Check non-overlap
        # Vectorized distance check
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        # We need dist >= min_dist for all i != j
        # np.fill_diagonal to ignore self
        np.fill_diagonal(dist, np.inf)
        np.fill_diagonal(min_dist, 0)
        
        if np.any(dist < min_dist - 1e-9):
            return False
        return True

    def make_initial_guess(seed):
        rng = np.random.RandomState(seed)
        # Perturb the grid
        x = best_x + rng.uniform(-0.05, 0.05, N)
        y = best_y + rng.uniform(-0.05, 0.05, N)
        r = best_r + rng.uniform(-0.01, 0.01, N)
        
        # Clip to valid bounds initially to avoid huge penalties
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        r = np.clip(r, 0.01, 0.4)
        
        return x, y, r

    def objective(params):
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        
        # Objective: Maximize sum of radii -> Minimize -sum(r)
        obj = -np.sum(r)
        
        # Penalty for overlaps
        # d_ij = sqrt((xi-xj)^2 + (yi-yj)^2)
        # constraint: d_ij >= ri + rj
        # penalty: max(0, ri + rj - d_ij)^2
        
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        
        # Ignore diagonal
        np.fill_diagonal(dist, 1.0) # Set to 1 to avoid div by 0 or issues, though we mask later
        np.fill_diagonal(min_dist, 0.0)
        
        # Smooth penalty
        # We only care about pairs i < j to avoid double counting
        # But for vectorized calc, we can sum over upper triangle
        # Or just sum all and divide by 2? No, max(0, ...) is not linear.
        # Let's use upper triangle indices
        i_idx, j_idx = np.triu_indices(N, k=1)
        d_ij = dist[i_idx, j_idx]
        r_sum = min_dist[i_idx, j_idx]
        
        violations = r_sum - d_ij
        # Square the positive violations
        pen = np.sum(np.maximum(0, violations)**2)
        
        return obj + 10.0 * pen

    def gradient(params):
        # Numerical gradient is too slow. L-BFGS-B can do finite differences.
        # But for speed, let's provide analytical if possible? 
        # For now, let scipy handle it with finite differences if we don't provide grad.
        # Actually, providing grad is much faster.
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        
        grad = np.zeros(3*N)
        
        # Gradient of -sum(r)
        grad[2*N:3*N] = -1.0
        
        # Gradient of penalty term
        # P = sum_{i<j} max(0, ri+rj - dij)^2
        # dP/dri = 2 * sum_{j!=i} max(0, ri+rj-dij) * sign(ri+rj-dij)
        # dP/dxi = -2 * sum_{j!=i} max(0, ri+rj-dij) * (xi-xj)/dij
        
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        dist[mask] = np.where(dist[mask] == 0, 1.0, dist[mask]) # Avoid div by 0
        
        violations = min_dist - dist
        active = np.maximum(0, violations)
        
        # Factor 2 * active
        # For radius gradient
        # dP/dri = 2 * sum_j active[i,j]
        # Note: active is symmetric.
        grad_r_pen = 2.0 * np.sum(active, axis=1)
        
        # For position gradient
        # dP/dxi = -2 * sum_j active[i,j] * (xi-xj)/dij
        # We need to be careful with dist=0
        # Let's compute term by term or vectorized
        # term_ij = active[i,j] * (xi-xj)/dij
        # dx_term = active * (dx / dist)
        # dy_term = active * (dy / dist)
        
        # Handle division by zero
        inv_dist = np.zeros_like(dist)
        inv_dist[mask] = 1.0 / dist[mask]
        
        dx_term = active * dx * inv_dist
        dy_term = active * dy * inv_dist
        
        grad_x_pen = -2.0 * np.sum(dx_term, axis=1)
        grad_y_pen = -2.0 * np.sum(dy_term, axis=1)
        
        # Combine
        grad[:N] = grad_x_pen * 10.0
        grad[N:2*N] = grad_y_pen * 10.0
        grad[2*N:3*N] = grad_r_pen * 10.0
        
        return grad

    def run_optimization(seed):
        x0, y0, r0 = make_initial_guess(seed)
        x0 = np.clip(x0, 0.01, 0.99)
        y0 = np.clip(y0, 0.01, 0.99)
        r0 = np.clip(r0, 0.01, 0.4)
        
        params0 = np.concatenate([x0, y0, r0])
        
        # Bounds
        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.0)) # x
        for i in range(N):
            bounds.append((0.0, 1.0)) # y
        for i in range(N):
            bounds.append((0.0, 0.5)) # r
            
        options = {'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        
        try:
            res = minimize(objective, params0, method='L-BFGS-B', jac=gradient, bounds=bounds, options=options)
            
            if res.success or res.status == 0:
                x_opt = res.x[:N]
                y_opt = res.x[N:2*N]
                r_opt = res.x[2*N:3*N]
                
                # Check validity
                if check_validity(x_opt, y_opt, r_opt):
                    val = np.sum(r_opt)
                    return x_opt, y_opt, r_opt, val
                else:
                    # Try to repair by shrinking radii slightly
                    # Find max violation
                    dx = x_opt[:, None] - x_opt[None, :]
                    dy = y_opt[:, None] - y_opt[None, :]
                    dist = np.sqrt(dx**2 + dy**2)
                    min_dist = r_opt[:, None] + r_opt[None, :]
                    np.fill_diagonal(dist, np.inf)
                    np.fill_diagonal(min_dist, 0)
                    
                    # Simple repair: scale down radii globally if invalid
                    # This is a heuristic.
                    for scale in [0.99, 0.98, 0.95]:
                        r_test = r_opt * scale
                        if check_validity(x_opt, y_opt, r_test):
                            return x_opt, y_opt, r_test, np.sum(r_test)
            return None
        except Exception:
            return None

    # Multi-start loop
    num_starts = 20
    for i in range(num_starts):
        if time.time() > DEADLINE:
            break
            
        res = run_optimization(i)
        if res is not None:
            x_opt, y_opt, r_opt, val = res
            if val > best_val:
                best_val = val
                best_x = x_opt
                best_y = y_opt
                best_r = r_opt
    
    # Final validity check and safety shrink
    if not check_validity(best_x, best_y, best_r):
        # Aggressive repair: shrink radii until valid
        for scale in np.linspace(0.99, 0.90, 10):
            r_test = best_r * scale
            if check_validity(best_x, best_y, r_test):
                best_r = r_test
                break
        else:
            # Fallback to seed
            best_x = np.array(grid_x)
            best_y = np.array(grid_y)
            best_r = np.array(grid_r)

    return np.column_stack((best_x, best_y, best_r))
