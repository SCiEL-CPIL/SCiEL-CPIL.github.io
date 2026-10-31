"""Rewritten program for circle packing in unit square."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 26

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    def check_validity(x, y, r, tol=1e-9):
        """Check if packing is valid (all constraints satisfied)."""
        if np.any(x < r - tol) or np.any(x > 1 - r + tol) or np.any(y < r - tol) or np.any(y > 1 - r + tol):
            return False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                if d < r[i] + r[j] - tol:
                    return False
        return True

    def force_directed_anneal(x0, y0, r0, n_steps=5000, T_start=0.1, T_end=0.001, max_radius=0.25):
        """Force-directed annealing to escape local optima and find better packings."""
        x = x0.copy()
        y = y0.copy()
        r = r0.copy()
        
        # Precompute which pairs are "close" for efficiency
        step_size = 0.01
        
        for step in range(n_steps):
            if time.time() > deadline:
                break
            
            # Cooling schedule
            T = T_start * (T_end / T_start) ** (step / n_steps)
            
            # Compute forces
            fx = np.zeros(N)
            fy = np.zeros(N)
            
            # Repulsion between overlapping circles
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx*dx + dy*dy) + 1e-10
                    req = r[i] + r[j]
                    if d < req:
                        # Repulsive force proportional to overlap
                        force = 10.0 * (req - d) / d
                        fx[i] += force * dx
                        fy[i] += force * dy
                        fx[j] -= force * dx
                        fy[j] -= force * dy
            
            # Attractive force to maximize radii (push circles apart)
            for i in range(N):
                # Boundary forces
                if x[i] < r[i]:
                    fx[i] += 5.0 * (r[i] - x[i])
                elif x[i] > 1 - r[i]:
                    fx[i] -= 5.0 * (x[i] - (1 - r[i]))
                if y[i] < r[i]:
                    fy[i] += 5.0 * (r[i] - y[i])
                elif y[i] > 1 - r[i]:
                    fy[i] -= 5.0 * (y[i] - (1 - r[i]))
            
            # Update positions with noise
            noise_x = np.random.normal(0, T, N)
            noise_y = np.random.normal(0, T, N)
            
            x_new = x + step_size * fx + noise_x
            y_new = y + step_size * fy + noise_y
            
            # Keep within bounds (soft constraint)
            x_new = np.clip(x_new, 0.01, 0.99)
            y_new = np.clip(y_new, 0.01, 0.99)
            
            # Try to grow radii
            r_new = r.copy()
            for i in range(N):
                # Max radius based on position
                max_r_i = min(x_new[i], 1 - x_new[i], y_new[i], 1 - y_new[i], max_radius)
                # Also limit by nearest neighbor
                min_dist = np.inf
                for j in range(N):
                    if i != j:
                        d = np.sqrt((x_new[i] - x_new[j])**2 + (y_new[i] - y_new[j])**2)
                        min_dist = min(min_dist, d - r_new[j])
                max_r_i = min(max_r_i, max(0.005, min_dist))
                r_new[i] = max(r[i], min(r_new[i] + 0.002, max_r_i))
            
            # Accept/reject based on energy
            # Simple acceptance: accept if valid or with probability exp(-dE/T)
            x = x_new
            y = y_new
            r = r_new
        
        # Final polish: ensure validity by shrinking if needed
        if not check_validity(x, y, r):
            # Binary search for valid scale
            lo, hi = 0.5, 1.0
            for _ in range(30):
                mid = (lo + hi) / 2
                if check_validity(x, y, r * mid):
                    lo = mid
                else:
                    hi = mid
            r = r * lo
        
        return x, y, r

    def local_optimize(x0, y0, r0, max_iter=200):
        """Run local optimization from initial guess."""
        z0 = np.concatenate([x0, y0, r0])
        
        def objective(z, penalty_weight=1000.0):
            x = z[:N]
            y = z[N:2*N]
            r = np.maximum(z[2*N:], 0.001)
            
            sum_r = np.sum(r)
            
            # Boundary violations
            v_bound = 0.0
            v_bound += np.sum(np.maximum(0, r - x)**2)
            v_bound += np.sum(np.maximum(0, r - (1 - x))**2)
            v_bound += np.sum(np.maximum(0, r - y)**2)
            v_bound += np.sum(np.maximum(0, r - (1 - y))**2)
            
            # Collision violations - vectorized
            x_diff = x[:, None] - x[None, :]
            y_diff = y[:, None] - y[None, :]
            dist_sq = x_diff**2 + y_diff**2
            dist = np.sqrt(dist_sq + 1e-15)
            req = r[:, None] + r[None, :]
            overlap = np.maximum(0, req - dist)
            v_coll = np.sum(overlap**2)
            
            return -sum_r + penalty_weight * (v_bound + v_coll)
        
        try:
            res = minimize(objective, z0, method='L-BFGS-B', 
                          options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})
            
            x_opt = res.x[:N]
            y_opt = res.x[N:2*N]
            r_opt = np.maximum(res.x[2*N:], 0.001)
            
            if check_validity(x_opt, y_opt, r_opt):
                return x_opt, y_opt, r_opt
            else:
                # Shrink to validity
                lo, hi = 0.5, 1.0
                for _ in range(30):
                    mid = (lo + hi) / 2
                    if check_validity(x_opt, y_opt, r_opt * mid):
                        lo = mid
                    else:
                        hi = mid
                r_opt = r_opt * lo
                return x_opt, y_opt, r_opt
        except Exception:
            return None

    def hexagonal_init(n_circles):
        """Create hexagonal packing initialization."""
        x = np.zeros(n_circles)
        y = np.zeros(n_circles)
        r = np.zeros(n_circles)
        
        cols = 5
        rows = int(np.ceil(n_circles / cols))
        
        for i in range(n_circles):
            row = i // cols
            col = i % cols
            x[i] = 0.1 + col * 0.16 + (row % 2) * 0.08
            y[i] = 0.1 + row * 0.17
            r[i] = 0.075
        
        # Ensure within bounds
        for i in range(n_circles):
            r[i] = min(r[i], min(x[i], 1-x[i], y[i], 1-y[i]) * 0.99)
        
        return x, y, r

    def random_init():
        """Random initialization with non-overlapping circles."""
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        # Place circles one by one, ensuring no overlap
        for i in range(N):
            for attempt in range(1000):
                ri = np.random.uniform(0.03, 0.12)
                xi = np.random.uniform(ri, 1-ri)
                yi = np.random.uniform(ri, 1-ri)
                
                # Check against already placed circles
                valid = True
                for j in range(i):
                    d = np.sqrt((xi - x[j])**2 + (yi - y[j])**2)
                    if d < ri + r[j] + 0.01:
                        valid = False
                        break
                
                if valid:
                    x[i] = xi
                    y[i] = yi
                    r[i] = ri
                    break
            else:
                # Fallback: place in corner
                x[i] = 0.1 + (i % 5) * 0.18
                y[i] = 0.1 + (i // 5) * 0.18
                r[i] = 0.06
        
        return x, y, r

    def grid_init():
        """Grid initialization."""
        side = 5
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        for i in range(N):
            row = i // side
            col = i % side
            x[i] = (col + 0.5) / side
            y[i] = (row + 0.5) / side
            r[i] = 0.5 / side * 0.9
        
        return x, y, r

    best_sum = 0.0
    best_result = None
    
    def update_best(x, y, r):
        nonlocal best_sum, best_result
        if check_validity(x, y, r):
            s = np.sum(r)
            if s > best_sum:
                best_sum = s
                best_result = np.column_stack((x, y, r))

    # Phase 1: Multi-start with force-directed annealing
    n_fd_restarts = 15
    for k in range(n_fd_restarts):
        if time.time() > deadline:
            break
        
        if k % 3 == 0:
            x0, y0, r0 = hexagonal_init(N)
            x0 += np.random.uniform(-0.01, 0.01, N)
            y0 += np.random.uniform(-0.01, 0.01, N)
        elif k % 3 == 1:
            x0, y0, r0 = random_init()
        else:
            x0, y0, r0 = grid_init()
            x0 += np.random.uniform(-0.01, 0.01, N)
            y0 += np.random.uniform(-0.01, 0.01, N)
        
        # Force-directed annealing
        x_fd, y_fd, r_fd = force_directed_anneal(x0, y0, r0, n_steps=3000)
        update_best(x_fd, y_fd, r_fd)
        
        # Local polish on the result
        result = local_optimize(x_fd, y_fd, r_fd, max_iter=150)
        if result is not None:
            update_best(*result)
    
    # Phase 2: Additional local optimization from best found so far
    if best_result is not None:
        x_best = best_result[:, 0]
        y_best = best_result[:, 1]
        r_best = best_result[:, 2]
        
        # Perturb and re-optimize
        for k in range(10):
            if time.time() > deadline:
                break
            
            x0 = x_best + np.random.uniform(-0.02, 0.02, N)
            y0 = y_best + np.random.uniform(-0.02, 0.02, N)
            r0 = r_best * np.random.uniform(0.95, 1.05, N)
            
            result = local_optimize(x0, y0, r0, max_iter=200)
            if result is not None:
                update_best(*result)
    
    # Fallback
    if best_result is None:
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_result
# EVOLVE-BLOCK-END
