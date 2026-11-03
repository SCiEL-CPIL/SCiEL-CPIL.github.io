import numpy as np
from scipy.optimize import minimize
import time

N = 26
DEADLINE = 140.0  # Safety margin from 150s limit

def construct():
    start_time = time.time()
    
    # Helper to check time
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
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
        
        np.fill_diagonal(dist, 1.0)
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

    def run_local_refinement(params0, max_iter=1000):
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
                options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            return res.x
        except Exception:
            return None

    def _max_radius_for_circle(x_i, y_i, x, y, r, i):
        """Calculate the maximum radius for circle i given others are fixed."""
        # Distance to boundaries
        r_bound = min(x_i, y_i, 1 - x_i, 1 - y_i)
        
        # Distance to other circles
        r_max = r_bound
        for j in range(N):
            if i == j:
                continue
            d = np.sqrt((x_i - x[j])**2 + (y_i - y[j])**2)
            r_max = min(r_max, d - r[j])
        
        return max(0.0, r_max)

    def _eval_packing(x, y, r):
        """Evaluate the packing, returning the sum of radii if valid, else -inf."""
        if check_validity(x, y, r):
            return np.sum(r)
        return -np.inf

    def _best_fit_local_search(x, y, r, max_iter=2000):
        """
        Discrete best-fit local search.
        1. Try to increase radius of each circle to its max possible.
        2. Try to move each circle to a better position (grid or random) that allows a larger radius or maintains radius.
        """
        x = x.copy()
        y = y.copy()
        r = r.copy()
        
        # First, maximize all radii
        for _ in range(50):
            changed = False
            for i in range(N):
                new_r = _max_radius_for_circle(x[i], y[i], x, y, r, i)
                if new_r > r[i] + 1e-10:
                    r[i] = new_r
                    changed = True
            if not changed:
                break
        
        # Now try to move circles
        # We use a set of candidate positions: current, grid points, and random
        # To keep it fast, we only try a few random positions per circle per iteration
        rng = np.random.RandomState(42)
        
        for it in range(max_iter):
            if time_left() < 10:
                break
                
            improved = False
            # Random order of circles
            order = rng.permutation(N)
            for i in order:
                current_val = np.sum(r) # We just want to maximize sum, so if any r increases, sum increases
                
                # Generate candidates
                # 1. Current position (with max radius)
                # 2. Nearby grid points
                # 3. Random points
                
                # Actually, a simpler approach: 
                # Try moving circle i to a random position, then maximize its radius.
                # If the new radius + sum of others > old radius + sum of others, accept.
                # Since others are fixed, we just need new_r_i > old_r_i.
                
                best_new_r = r[i]
                best_new_x = x[i]
                best_new_y = y[i]
                
                # Try a few random positions
                for _ in range(10):
                    nx = rng.uniform(0, 1)
                    ny = rng.uniform(0, 1)
                    nr = _max_radius_for_circle(nx, ny, x, y, r, i)
                    if nr > best_new_r + 1e-10:
                        best_new_r = nr
                        best_new_x = nx
                        best_new_y = ny
                
                # Also try the "best fit" on a fine grid around current position? 
                # Maybe just random is enough for now, or add a small perturbation.
                # Let's add a few perturbations
                for _ in range(5):
                    nx = x[i] + rng.uniform(-0.1, 0.1)
                    ny = y[i] + rng.uniform(-0.1, 0.1)
                    nx = np.clip(nx, 0, 1)
                    ny = np.clip(ny, 0, 1)
                    nr = _max_radius_for_circle(nx, ny, x, y, r, i)
                    if nr > best_new_r + 1e-10:
                        best_new_r = nr
                        best_new_x = nx
                        best_new_y = ny

                if best_new_r > r[i] + 1e-10:
                    x[i] = best_new_x
                    y[i] = best_new_y
                    r[i] = best_new_r
                    improved = True
            
            if not improved:
                break
                
        return x, y, r

    def _gradient_polish(x, y, r):
        """Use gradient descent to fine-tune positions and radii."""
        params = np.concatenate([x, y, r])
        refined = run_local_refinement(params, max_iter=500)
        if refined is not None:
            x_new = refined[:N]
            y_new = refined[N:2*N]
            r_new = refined[2*N:3*N]
            if check_validity(x_new, y_new, r_new):
                return x_new, y_new, r_new
        return x, y, r

    # Strategy:
    # 1. Start with grid
    # 2. Run best-fit local search
    # 3. Run gradient polish
    # 4. Repeat with random restarts if time permits

    # Initial grid
    x_cur = best_x.copy()
    y_cur = best_y.copy()
    r_cur = best_r.copy()

    # Refine grid
    x_cur, y_cur, r_cur = _best_fit_local_search(x_cur, y_cur, r_cur)
    x_cur, y_cur, r_cur = _gradient_polish(x_cur, y_cur, r_cur)
    
    val = _eval_packing(x_cur, y_cur, r_cur)
    if val > best_val:
        best_x, best_y, best_r, best_val = x_cur, y_cur, r_cur, val

    # Random restarts
    rng = np.random.RandomState(123)
    while time_left() > 20:
        # Generate random initial positions
        x0 = rng.uniform(0.1, 0.9, N)
        y0 = rng.uniform(0.1, 0.9, N)
        # Start with small radii to avoid overlap
        r0 = rng.uniform(0.01, 0.03, N)
        
        # Run best-fit local search
        x_new, y_new, r_new = _best_fit_local_search(x0, y0, r0, max_iter=500)
        
        # Run gradient polish
        x_new, y_new, r_new = _gradient_polish(x_new, y_new, r_new)
        
        val = _eval_packing(x_new, y_new, r_new)
        if val > best_val:
            best_x, best_y, best_r, best_val = x_new, y_new, r_new, val

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
