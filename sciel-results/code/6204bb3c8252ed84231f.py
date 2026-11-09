import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    best_score = 0.0
    best_config = None

    def validity_penalty(x, y, r, tol=1e-9):
        """
        Returns 0 if valid, large value if invalid.
        """
        n = len(r)
        if np.any(r < -tol):
            return 1e10
        if np.any(x - r < -tol) or np.any(y - r < -tol):
            return 1e10
        if np.any(x + r > 1 + tol) or np.any(y + r > 1 + tol):
            return 1e10
        
        # Check overlaps
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist_sq = dx*dx + dy*dy
        r_sum = r[:, None] + r[None, :]
        
        # We need dist >= r_sum for all i != j
        # dist_sq >= (r_sum - tol)^2
        mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(mask, False)
        
        if np.any(dist_sq[mask] < (r_sum[mask] - tol)**2):
            return 1e10
        return 0.0

    def repair_config(x, y, r):
        """
        Attempts to make a configuration valid by shrinking radii and moving points.
        """
        # 1. Clamp positions to [0,1]
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        
        # 2. Initial radius clamp based on boundaries
        r_max_bound = np.minimum(
            np.minimum(x, 1 - x),
            np.minimum(y, 1 - y)
        )
        r = np.minimum(r, r_max_bound)
        r = np.maximum(r, 0.0)
        
        # 3. Iterative shrink if overlaps persist
        for _ in range(100):
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d_sq = dx**2 + dy**2
            np.fill_diagonal(d_sq, np.inf)
            
            r_sum = r[:, None] + r[None, :]
            # Overlap if dist < r_sum
            # We want to shrink r such that dist >= r_sum
            overlaps = d_sq < (r_sum - 1e-9)**2
            np.fill_diagonal(overlaps, False)
            
            if not np.any(overlaps):
                return x, y, r
            
            # Shrink all radii slightly to resolve overlaps
            # More aggressive shrink
            r *= 0.9
            
        return x, y, r

    def maximize_radii(x, y, r, max_iter=100):
        """
        Greedily increase radii as much as possible while maintaining validity.
        """
        r_curr = r.copy()
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            improved = False
            # Try to increase each radius slightly
            for i in range(N):
                # Max possible increase for i is bounded by boundary and neighbors
                # Check boundary
                max_r_i = min(x[i], 1 - x[i], y[i], 1 - y[i])
                
                # Check neighbors
                for j in range(N):
                    if i == j: continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    # r_i + r_j <= dist  => r_i <= dist - r_j
                    max_r_i = min(max_r_i, dist - r_curr[j])
                
                # If we can increase r_i, do so
                if max_r_i > r_curr[i] + 1e-8:
                    # Binary search or step up
                    # We want to set r_curr[i] as high as possible
                    # But increasing one might block others? No, we check validity.
                    # However, simply setting to max_r_i might be too aggressive if it causes
                    # other circles to be unable to grow? No, we only check current validity.
                    # But wait, if we set r_i to max, it might push against j, but that's fine
                    # as long as they don't overlap.
                    
                    # Let's try to jump to max_r_i
                    test_r = r_curr.copy()
                    test_r[i] = max_r_i
                    if validity_penalty(x, y, test_r) == 0:
                        r_curr[i] = max_r_i
                        improved = True
                    else:
                        # If max_r_i is invalid, it means our calculation of max_r_i was wrong?
                        # Or numerical issues.
                        # Let's try a smaller step.
                        step = 0.5 * (max_r_i - r_curr[i])
                        test_r[i] = r_curr[i] + step
                        if validity_penalty(x, y, test_r) == 0:
                            r_curr[i] = test_r[i]
                            improved = True
            
            if not improved:
                break
                
        return r_curr

    def objective(z):
        """
        Objective function for L-BFGS-B.
        z is a vector of length 3*N: [x_1, y_1, r_1, ..., x_N, y_N, r_N]
        We want to maximize sum(r), so we minimize -sum(r).
        """
        if time.time() > deadline:
            return 1e10
            
        x = z[0::3]
        y = z[1::3]
        r = z[2::3]
        
        # Base objective: -sum(r)
        obj = -np.sum(r)
        
        # Boundary penalties
        w_bound = 1000.0
        
        viol_x_left = np.maximum(0, r - x)
        viol_x_right = np.maximum(0, x + r - 1)
        viol_y_bottom = np.maximum(0, r - y)
        viol_y_top = np.maximum(0, y + r - 1)
        
        obj += w_bound * (np.sum(viol_x_left**2) + np.sum(viol_x_right**2) + 
                          np.sum(viol_y_bottom**2) + np.sum(viol_y_top**2))
        
        # Overlap penalties
        w_overlap = 1000.0
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(d, np.inf)
        
        r_sum = r[:, None] + r[None, :]
        
        overlap = np.maximum(0, r_sum - d)
        np.fill_diagonal(overlap, 0)
        
        obj += w_overlap * np.sum(overlap**2)
        
        return obj

    def optimize_l_bfgs(x, y, r, max_iter=1000):
        """
        Uses L-BFGS-B to optimize positions and radii simultaneously.
        """
        z0 = np.concatenate((x, y, r))
        
        bounds = []
        for i in range(N):
            bounds.append((0, 1))  # x
            bounds.append((0, 1))  # y
            bounds.append((0.001, 0.5))  # r
            
        try:
            res = minimize(
                objective, 
                z0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-5}
            )
            if res.success or res.fun < 1e10:
                z_opt = res.x
                x_opt = z_opt[0::3]
                y_opt = z_opt[1::3]
                r_opt = z_opt[2::3]
                
                # Ensure strict validity and repair if needed
                x_opt, y_opt, r_opt = repair_config(x_opt, y_opt, r_opt)
                
                # Final radius maximization
                r_opt = maximize_radii(x_opt, y_opt, r_opt, max_iter=50)
                
                return x_opt, y_opt, r_opt
        except Exception:
            pass
            
        return x, y, r

    def local_search(x, y, r, max_iter=200):
        """
        Simple local search to escape local minima.
        """
        best_x, best_y, best_r = x.copy(), y.copy(), r.copy()
        best_val = np.sum(best_r)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
                
            idx = np.random.randint(N)
            cand_x = best_x.copy()
            cand_y = best_y.copy()
            cand_r = best_r.copy()
            
            # Perturb position
            scale = 0.02
            cand_x[idx] += np.random.normal(0, scale)
            cand_y[idx] += np.random.normal(0, scale)
            cand_x[idx] = np.clip(cand_x[idx], 0, 1)
            cand_y[idx] = np.clip(cand_y[idx], 0, 1)
            
            # Perturb radius
            cand_r[idx] += np.random.normal(0, scale * 0.5)
            cand_r[idx] = np.clip(cand_r[idx], 0.01, 0.5)
            
            # Quick validity check
            if validity_penalty(cand_x, cand_y, cand_r) == 0:
                val = np.sum(cand_r)
                if val > best_val:
                    best_x, best_y, best_r = cand_x, cand_y, cand_r
                    best_val = val
                    
        return best_x, best_y, best_r

    num_restarts = 40

    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        np.random.seed(42 + restart * 13)
        
        # Initialization
        # Strategy: Grid with jitter
        side = int(np.ceil(np.sqrt(N)))
        xs = []
        ys = []
        rs = []
        
        # Fill grid
        for i in range(side):
            for j in range(side):
                if len(xs) >= N:
                    break
                x = (i + 0.5) / side + np.random.uniform(-0.1, 0.1)
                y = (j + 0.5) / side + np.random.uniform(-0.1, 0.1)
                x = np.clip(x, 0.05, 0.95)
                y = np.clip(y, 0.05, 0.95)
                xs.append(x)
                ys.append(y)
                # Initial radius estimate: half the grid cell size
                rs.append(0.5 / side * 0.8)
                
        # If grid didn't fill, fill random
        while len(xs) < N:
            xs.append(np.random.uniform(0.1, 0.9))
            ys.append(np.random.uniform(0.1, 0.9))
            rs.append(0.05)
            
        x = np.array(xs)
        y = np.array(ys)
        r = np.array(rs)
        
        # Phase 1: L-BFGS-B Optimization
        x, y, r = optimize_l_bfgs(x, y, r, max_iter=500)
        
        # Phase 2: Local search to refine
        x, y, r = local_search(x, y, r, max_iter=100)
        
        # Phase 3: Final maximize radii
        r = maximize_radii(x, y, r, max_iter=50)
        
        # Repair if necessary
        x, y, r = repair_config(x, y, r)
        
        # Final Check
        if validity_penalty(x, y, r) == 0:
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_config = np.column_stack((x, y, r))
                
    if best_config is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_config
