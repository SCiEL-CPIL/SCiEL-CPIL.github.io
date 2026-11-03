import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 26
DEADLINE = 110.0  # Safety margin from 120s limit

def construct():
    start_time = time.time()
    
    # Baseline: Grid packing
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
    
    def check_validity(x, y, r, tol=1e-9):
        # Check boundary
        if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
            return False
        if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
            return False
        if np.any(r < -tol):
            return False
        
        # Check non-overlap
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        
        # Ignore self
        np.fill_diagonal(dist, np.inf)
        np.fill_diagonal(min_dist, 0)
        
        if np.any(dist < min_dist - tol):
            return False
        return True

    def get_penalty(params, penalty_weight=100.0):
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        
        # Objective: Minimize -sum(r) + penalty
        obj = -np.sum(r)
        
        # Penalty for overlaps
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        
        np.fill_diagonal(dist, 1.0) # Avoid div by zero in gradient if needed, though DE uses finite diff
        np.fill_diagonal(min_dist, 0.0)
        
        i_idx, j_idx = np.triu_indices(N, k=1)
        d_ij = dist[i_idx, j_idx]
        r_sum = min_dist[i_idx, j_idx]
        
        violations = r_sum - d_ij
        pen = np.sum(np.maximum(0, violations)**2)
        
        return obj + penalty_weight * pen

    def get_penalty_grad(params, penalty_weight=100.0):
        x = params[:N]
        y = params[N:2*N]
        r = params[2*N:3*N]
        
        grad = np.zeros(3*N)
        grad[2*N:3*N] = -1.0
        
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        
        mask = ~np.eye(N, dtype=bool)
        dist[mask] = np.where(dist[mask] == 0, 1e-10, dist[mask])
        
        violations = min_dist - dist
        active = np.maximum(0, violations)
        
        # Gradient of penalty
        # dP/dr_i = 2 * sum_j active[i,j]
        grad_r_pen = 2.0 * np.sum(active, axis=1)
        
        # dP/dx_i = -2 * sum_j active[i,j] * (x_i - x_j) / d_ij
        inv_dist = np.zeros_like(dist)
        inv_dist[mask] = 1.0 / dist[mask]
        
        dx_term = active * dx * inv_dist
        dy_term = active * dy * inv_dist
        
        grad_x_pen = -2.0 * np.sum(dx_term, axis=1)
        grad_y_pen = -2.0 * np.sum(dy_term, axis=1)
        
        grad[:N] += penalty_weight * grad_x_pen
        grad[N:2*N] += penalty_weight * grad_y_pen
        grad[2*N:3*N] += penalty_weight * grad_r_pen
        
        return grad

    def run_local_refinement(params0):
        """Refine a candidate using L-BFGS-B"""
        bounds = []
        for i in range(N):
            bounds.append((0.0, 1.0)) # x
        for i in range(N):
            bounds.append((0.0, 1.0)) # y
        for i in range(N):
            bounds.append((0.0, 0.5)) # r
            
        try:
            res = minimize(
                get_penalty, 
                params0, 
                method='L-BFGS-B', 
                jac=get_penalty_grad, 
                bounds=bounds, 
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            return res.x
        except Exception:
            return None

    # Strategy:
    # 1. Differential Evolution to find a good global configuration
    # 2. Local refinement with L-BFGS-B
    
    # Define bounds for DE
    bounds_de = []
    for i in range(N):
        bounds_de.append((0.0, 1.0)) # x
    for i in range(N):
        bounds_de.append((0.0, 1.0)) # y
    for i in range(N):
        bounds_de.append((0.0, 0.4)) # r (upper bound 0.4 is generous)

    # Run DE
    # popsize=15, maxiter=50 is a reasonable start. 
    # We need to be careful with time. 
    # DE is stochastic, so we run it once with a decent budget, then refine.
    
    try:
        de_result = differential_evolution(
            get_penalty, 
            bounds_de, 
            seed=42, 
            maxiter=40, 
            popsize=10, 
            tol=1e-6,
            init='latinhypercube',
            updateinterval=10,
            polish=False # We will do our own polish
        )
        
        if de_result.success:
            best_params_de = de_result.x
            # Refine with L-BFGS-B
            refined_params = run_local_refinement(best_params_de)
            
            if refined_params is not None:
                x_opt = refined_params[:N]
                y_opt = refined_params[N:2*N]
                r_opt = refined_params[2*N:3*N]
                
                # Check validity and repair if necessary
                if check_validity(x_opt, y_opt, r_opt):
                    val = np.sum(r_opt)
                    if val > best_val:
                        best_x, best_y, best_r, best_val = x_opt, y_opt, r_opt, val
                else:
                    # Try to repair by shrinking radii
                    # Find the maximum violation
                    dx = x_opt[:, None] - x_opt[None, :]
                    dy = y_opt[:, None] - y_opt[None, :]
                    dist = np.sqrt(dx**2 + dy**2)
                    min_dist = r_opt[:, None] + r_opt[None, :]
                    np.fill_diagonal(dist, np.inf)
                    np.fill_diagonal(min_dist, 0)
                    
                    max_violation = np.max(np.maximum(0, min_dist - dist))
                    
                    # If violation is small, shrink radii slightly
                    # This is a heuristic repair
                    for scale in [0.99, 0.98, 0.95, 0.90]:
                        r_test = r_opt * scale
                        if check_validity(x_opt, y_opt, r_test):
                            val = np.sum(r_test)
                            if val > best_val:
                                best_x, best_y, best_r, best_val = x_opt, y_opt, r_test, val
                            break
                            
    except Exception as e:
        # If DE fails, we rely on the grid or try a simple random start + L-BFGS
        pass

    # If time permits, try a few random restarts with L-BFGS
    if time.time() < DEADLINE - 20:
        rng = np.random.RandomState(43)
        for i in range(5):
            if time.time() > DEADLINE - 10:
                break
            
            # Random initial guess, but keep radii small to avoid massive penalties
            x0 = rng.uniform(0.1, 0.9, N)
            y0 = rng.uniform(0.1, 0.9, N)
            r0 = rng.uniform(0.01, 0.05, N)
            
            params0 = np.concatenate([x0, y0, r0])
            
            refined_params = run_local_refinement(params0)
            if refined_params is not None:
                x_opt = refined_params[:N]
                y_opt = refined_params[N:2*N]
                r_opt = refined_params[2*N:3*N]
                
                if check_validity(x_opt, y_opt, r_opt):
                    val = np.sum(r_opt)
                    if val > best_val:
                        best_x, best_y, best_r, best_val = x_opt, y_opt, r_opt, val

    # Final safety check
    if not check_validity(best_x, best_y, best_r):
        # Aggressive repair
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
