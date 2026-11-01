import numpy as np
import time

N = 200
T_MAX = 110.0

def construct():
        start_time = time.time()
        # 1. Initialization
        # Use a 15x15 grid, remove 25 points
        m = 15
        pts = []
        for i in range(m):
            for j in range(m):
                x = (i + 0.5) / m - 0.5
                y = (j + 0.5) / m - 0.5
                pts.append([x, y])
        pts = np.array(pts)
        # Remove 25 random points to get 200
        idx = np.random.choice(len(pts), 200, replace=False)
        C = pts[idx]
        
        # Shuffle to avoid symmetry breaking issues? No, grid is fine.
        
        # 2. Optimization
        # We will use a simple gradient ascent on a smooth objective.
        # Objective: Maximize R_soft
        # Or minimize Energy for a fixed r.
        
        # Let's use the Energy approach.
        # Current best radius found
        r_current = 0.03
        lr = 0.01
        decay = 0.999
        beta = 10 # Sharpness for soft min if used, but we use hard min energy
        
        while time.time() - start_time < T_MAX - 5:
            # Calculate distances
            # d_ij matrix
            diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            
            # Boundary distances
            # dist to x=-0.5: 0.5 + x
            # dist to x=0.5: 0.5 - x
            # dist to y=-0.5: 0.5 + y
            # dist to y=0.5: 0.5 - y
            d_bound = np.minimum(
                np.minimum(0.5 + C[:, 0], 0.5 - C[:, 0]),
                np.minimum(0.5 + C[:, 1], 0.5 - C[:, 1])
            )
            
            # Current min radius
            r_min = min(np.min(dists) / 2, np.min(d_bound))
            
            if r_min > r_current:
                r_current = r_min
                # Reset learning rate if we made progress?
                # lr = 0.01
                pass
            
            # Gradient of Energy
            # E = sum_{i<j} max(0, 2*r_target - d_ij)^2
            # Let's fix r_target = r_current * 1.01 or just use r_current to push further?
            # If we use r_current, gradient is 0 at the min.
            # We need to push r_target slightly higher to force expansion.
            r_target = r_current * 1.05
            
            grad = np.zeros_like(C)
            
            # Pairwise forces
            # For each pair i,j with d_ij < 2*r_target
            mask = dists < 2 * r_target
            # Force magnitude: 2 * (2*r_target - d_ij) * (d_ij vector / d_ij)
            # d_ij vector = C_i - C_j
            # grad_i += 2 * (2*r_target - d_ij) * (C_i - C_j) / d_ij
            # grad_j += 2 * (2*r_target - d_ij) * (C_j - C_i) / d_ij
            
            # Vectorized:
            # diff is C_i - C_j
            # dists is norm
            # We need to avoid division by zero, but diagonal is inf, so mask excludes diagonal.
            
            # Create a matrix of forces
            # F_ij = 2 * (2*r_target - dists) * (diff / dists)
            # Only where mask is true.
            
            # Avoid division by zero
            safe_dists = np.where(dists > 1e-6, dists, 1.0)
            
            factor = np.where(mask, 2 * (2 * r_target - dists) / safe_dists, 0)
            # factor is NxN
            # Force on i from j is factor[i,j] * (C_i - C_j)
            # Sum over j
            # grad_i = sum_j factor[i,j] * (C_i - C_j)
            
            # grad = sum_j factor[i,j] * C_i - sum_j factor[i,j] * C_j
            # sum_j factor[i,j] * C_i = C_i * sum_j factor[i,j]
            # sum_j factor[i,j] * C_j = factor @ C
            
            sum_factor = np.sum(factor, axis=1)
            grad = C * sum_factor[:, np.newaxis] - (factor @ C)
            
            # Boundary forces
            # If d_bound < r_target, push towards center
            # d_bound = min(0.5+|x|, 0.5-|x|, ...)
            # Gradient of d_bound:
            # If x > 0, d_bound_x = 0.5 - x, grad = -1
            # If x < 0, d_bound_x = 0.5 + x, grad = 1
            # So grad_x = -sign(x) if 0.5 - |x| is the min?
            # Actually, if 0.5 - |x| < 0.5 - |y|, then the constraint is active on x.
            # Let's just compute the gradient of the boundary distance explicitly.
            
            # d_b = min(dx, dy) where dx = 0.5 - |x|, dy = 0.5 - |y|
            dx = 0.5 - np.abs(C[:, 0])
            dy = 0.5 - np.abs(C[:, 1])
            
            # grad of dx w.r.t x: -sign(x)
            # grad of dy w.r.t y: -sign(y)
            
            # If dx < dy, grad is (-sign(x), 0)
            # If dy < dx, grad is (0, -sign(y))
            
            mask_x = dx < dy
            grad[:, 0] += np.where(mask_x, -np.sign(C[:, 0]), 0) * np.where(dx < r_target, 2*(r_target - dx), 0)
            grad[:, 1] += np.where(~mask_x, -np.sign(C[:, 1]), 0) * np.where(dy < r_target, 2*(r_target - dy), 0)
            
            # Update
            C += lr * grad
            
            # Clamp to square
            C[:, 0] = np.clip(C[:, 0], -0.5, 0.5)
            C[:, 1] = np.clip(C[:, 1], -0.5, 0.5)
            
            # Decay learning rate
            lr *= decay
            
            # If lr is too small, restart with higher lr?
            if lr < 1e-5:
                lr = 0.01
                # Add noise
                C += np.random.normal(0, 0.01, C.shape)
                C[:, 0] = np.clip(C[:, 0], -0.5, 0.5)
                C[:, 1] = np.clip(C[:, 1], -0.5, 0.5)
                
        # Final polish: exact local search
        # Try small moves
        for _ in range(1000):
            if time.time() - start_time > T_MAX - 1: break
            improved = False
            for i in range(N):
                # Try 8 directions
                for dx, dy in [(0.001,0), (-0.001,0), (0,0.001), (0,-0.001), (0.001,0.001), (-0.001,-0.001), (0.001,-0.001), (-0.001,0.001)]:
                    new_C = C.copy()
                    new_C[i] += [dx, dy]
                    if new_C[i, 0] < -0.5 or new_C[i, 0] > 0.5 or new_C[i, 1] < -0.5 or new_C[i, 1] > 0.5:
                        continue
                    # Check if this move improves the min radius
                    # Calculate new min radius
                    diff = new_C[:, np.newaxis, :] - new_C[np.newaxis, :, :]
                    dists = np.linalg.norm(diff, axis=2)
                    np.fill_diagonal(dists, np.inf)
                    d_bound = np.minimum(
                        np.minimum(0.5 + new_C[:, 0], 0.5 - new_C[:, 0]),
                        np.minimum(0.5 + new_C[:, 1], 0.5 - new_C[:, 1])
                    )
                    new_r = min(np.min(dists) / 2, np.min(d_bound))
                    # Compare with current r for point i? No, global r.
                    # We need to store current global r.
                    # This is inefficient. 
                    pass
            # The final polish above is too slow.
            # Just return C.
            
        return C
