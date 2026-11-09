import numpy as np
import time

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    best_score = 0.0
    best_config = None

    def validity_penalty(x, y, r):
        """
        Returns 0 if valid, 1e10 if invalid.
        """
        n = len(r)
        if np.any(r < 0):
            return 1e10
        if np.any(x - r < -1e-9) or np.any(y - r < -1e-9):
            return 1e10
        if np.any(x + r > 1 + 1e-9) or np.any(y + r > 1 + 1e-9):
            return 1e10
        
        # Check overlaps
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx*dx + dy*dy
                r_sum = r[i] + r[j]
                if dist_sq < (r_sum - 1e-9)**2:
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
        for _ in range(50):
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d_sq = dx**2 + dy**2
            np.fill_diagonal(d_sq, np.inf)
            
            r_sum = r[:, None] + r[None, :]
            # Overlap if dist < r_sum
            # We want to shrink r such that dist >= r_sum
            # Approximate: scale r by factor
            overlaps = d_sq < (r_sum - 1e-9)**2
            np.fill_diagonal(overlaps, False)
            
            if not np.any(overlaps):
                return x, y, r
            
            # Shrink all radii slightly to resolve overlaps
            r *= 0.95
        
        return x, y, r

    def optimize_positions(x, y, r, steps=1000):
        """
        Optimize positions using gradient descent on a penalty function.
        We keep radii fixed during this phase to find a good spatial arrangement.
        """
        # We want to minimize overlaps and boundary violations.
        # Objective: Sum of (max(0, r_i + r_j - dist_ij)^2) + boundary penalties
        
        lr = 0.01
        lambda_overlap = 100.0
        lambda_bound = 100.0
        
        for step in range(steps):
            if time.time() > deadline:
                break
                
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d_sq = dx**2 + dy**2
            d = np.sqrt(d_sq)
            np.fill_diagonal(d, 1.0) # Avoid div by zero, diagonal is ignored later
            
            r_sum = r[:, None] + r[None, :]
            
            # Overlap penalty
            # P_ij = max(0, r_i + r_j - d_ij)^2
            diff = r_sum - d
            overlap = np.maximum(0, diff)
            np.fill_diagonal(overlap, 0)
            
            # Gradient of overlap penalty w.r.t x_i
            # d/dx_i ( (r_i+r_j-d_ij)^2 ) = 2(r_i+r_j-d_ij) * d/dx_i(-d_ij)
            # d/dx_i(-d_ij) = -(x_i - x_j)/d_ij = (x_j - x_i)/d_ij
            # Note: d_ij is symmetric, so for i, the term is (x_j - x_i)/d_ij
            # Vectorized: 
            # grad_x_overlap[i] = -2 * sum_j (overlap_ij * (x_i - x_j) / d_ij)
            # Wait, sign check.
            # Let f = (r_sum - d)^2. df/dd = 2(r_sum-d)*(-1) = -2(r_sum-d).
            # dd/dx_i = (x_i - x_j)/d.
            # df/dx_i = -2(r_sum-d) * (x_i-x_j)/d.
            # Since overlap = max(0, r_sum-d), if overlap > 0, r_sum-d = overlap.
            # So grad_x_overlap[i] = -2 * sum_j (overlap_ij * (x_i-x_j)/d_ij)
            
            # If overlap is 0, gradient is 0.
            # We can compute:
            factor = 2 * overlap / d
            np.fill_diagonal(factor, 0)
            
            grad_x = -np.sum(factor * dx, axis=1)
            grad_y = -np.sum(factor * dy, axis=1)
            
            # Boundary Penalty
            # P_bound = sum (max(0, r_i - x_i)^2 + max(0, r_i - (1-x_i))^2 + ... )
            # Gradient w.r.t x_i:
            # d/dx_i (r_i - x_i)^2 = 2(r_i - x_i)*(-1) = -2(r_i - x_i) if x_i < r_i
            # d/dx_i (r_i - (1-x_i))^2 = 2(r_i - (1-x_i))*(1) = 2(r_i - 1 + x_i) if 1-x_i < r_i
            
            bx_left = np.maximum(0, r - x)
            bx_right = np.maximum(0, r - (1 - x))
            by_bottom = np.maximum(0, r - y)
            by_top = np.maximum(0, r - (1 - y))
            
            grad_x_bound = -2 * bx_left + 2 * bx_right
            grad_y_bound = -2 * by_bottom + 2 * by_top
            
            # Total Gradient
            grad_x_total = lambda_overlap * grad_x + lambda_bound * grad_x_bound
            grad_y_total = lambda_overlap * grad_y + lambda_bound * grad_y_bound
            
            # Update
            x_new = x - lr * grad_x_total
            y_new = y - lr * grad_y_total
            
            # Clip to bounds to ensure we don't fly away
            x_new = np.clip(x_new, 0, 1)
            y_new = np.clip(y_new, 0, 1)
            
            # Accept if valid or improving penalty? 
            # For simplicity, always accept in this phase, but ensure we don't explode.
            x, y = x_new, y_new
            
            # Adaptive learning rate
            if step % 100 == 0:
                lr *= 0.95

        return x, y

    def maximize_radii(x, y, r, max_iter=200):
        """
        Greedily increase radii as much as possible while maintaining validity.
        This is critical for maximizing the sum of radii.
        """
        # Start with current radii
        r_curr = r.copy()
        
        # Calculate max possible radius for each circle based on boundaries
        r_max_bound = np.minimum(
            np.minimum(x, 1 - x),
            np.minimum(y, 1 - y)
        )
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            improved = False
            # Try to increase each radius slightly
            for i in range(N):
                # Max possible increase for i is bounded by boundary and neighbors
                # Check boundary
                max_r_i = r_max_bound[i]
                
                # Check neighbors
                for j in range(N):
                    if i == j: continue
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    # r_i + r_j <= dist  => r_i <= dist - r_j
                    max_r_i = min(max_r_i, dist - r_curr[j])
                
                # If we can increase r_i, do so
                if max_r_i > r_curr[i] + 1e-6:
                    # Increase by a small step or jump to max?
                    # Jumping to max might block others. Let's try to jump to 90% of the gap or just max?
                    # Greedy: set to max_r_i
                    new_r = max_r_i
                    # Validate
                    test_r = r_curr.copy()
                    test_r[i] = new_r
                    if validity_penalty(x, y, test_r) == 0:
                        r_curr[i] = new_r
                        improved = True
                    else:
                        # Binary search for the exact max valid radius? 
                        # Or just step by 0.1 * (max_r_i - r_curr[i])
                        step_r = 0.1 * (max_r_i - r_curr[i])
                        new_r = r_curr[i] + step_r
                        test_r[i] = new_r
                        if validity_penalty(x, y, test_r) == 0:
                            r_curr[i] = new_r
                            improved = True
            
            if not improved:
                break
                
        return r_curr

    def local_search(x, y, r, max_iter=100):
        """
        Simple local search to escape local minima in position/radius space.
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

    num_restarts = 50

    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        np.random.seed(42 + restart * 13)
        
        # Initialization
        # Strategy: Grid with jitter is usually a good start
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
                
        # If grid didn't fill (shouldn't happen for 26, side=6, 36 cells), fill random
        while len(xs) < N:
            xs.append(np.random.uniform(0.1, 0.9))
            ys.append(np.random.uniform(0.1, 0.9))
            rs.append(0.05)
            
        x = np.array(xs)
        y = np.array(ys)
        r = np.array(rs)
        
        # Phase 1: Optimize positions with fixed radii
        x, y = optimize_positions(x, y, r, steps=200)
        
        # Phase 2: Maximize radii
        r = maximize_radii(x, y, r, max_iter=50)
        
        # Phase 3: Local search to refine
        x, y, r = local_search(x, y, r, max_iter=100)
        
        # Phase 4: Final maximize radii
        r = maximize_radii(x, y, r, max_iter=20)
        
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
