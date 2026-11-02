import numpy as np
import time

N = 200

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 110.0  # Safety margin for 120s limit
    
    # 1. Initialization: Hexagonal lattice
    # Generate a hexagonal grid and select N points
    # Hex lattice spacing s. We want to fill roughly a 1x1 square.
    # Area of hex lattice point is s^2 * sqrt(3)/2.
    # N points -> Area ~ N * s^2 * sqrt(3)/2.
    # If we fit in 1x1, s ~ sqrt(2/(N*sqrt(3))).
    
    def hex_init():
        # Create a grid of hex points
        s = 1.0 / np.sqrt(N * np.sqrt(3) / 2) # Approximate spacing
        # Actually, let's just generate a large grid and pick the ones in the center
        # Or simpler:
        # Row 0: 0, s, 2s...
        # Row 1: s/2, 1.5s, 2.5s...
        
        points = []
        # Estimate rows needed
        rows = int(np.sqrt(N / (np.sqrt(3)/2))) + 2
        cols = rows + 2
        
        for r in range(-rows//2, rows//2 + 1):
            for c in range(-cols//2, cols//2 + 1):
                x = c * s + (r % 2) * s / 2
                y = r * s * np.sqrt(3) / 2
                points.append([x, y])
        
        points = np.array(points)
        # Center the points
        center = points.mean(axis=0)
        points = points - center
        
        # Scale to fit in roughly [-0.4, 0.4] to leave room for optimization
        # Current extent is roughly rows*s*sqrt(3)/2
        extent = np.max(np.abs(points))
        scale = 0.45 / extent
        points = points * scale
        
        # Randomly select N points, or just take the first N?
        # Taking first N might be biased. Let's shuffle.
        idx = np.random.choice(len(points), N, replace=False)
        return points[idx]

    # We will try a few different initializations or just one good one?
    # Given the time budget, let's do one robust run with adaptive parameters.
    
    c = hex_init()
    
    # Ensure finite and inside
    c = np.clip(c, -0.49, 0.49)
    
    best_score = -1.0
    best_c = c.copy()
    
    # Optimization parameters
    beta_start = 10.0
    beta_end = 1000.0
    steps = 2000 # Max steps, but will break on time
    
    # Precompute pairs for efficiency?
    # N=200 -> ~20k pairs. Computing distances each step is O(N^2).
    # 2000 steps * 20000 ops = 40M ops. This is fast in NumPy.
    
    for step in range(steps):
        if time.time() - start_time > deadline:
            break
            
        # Anneal beta
        progress = step / steps
        beta = beta_start * (beta_end / beta_start) ** progress
        
        # Compute distances
        # Pairwise
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf) # Ignore self
        
        # Wall distances
        # dist to x=0.5, x=-0.5, y=0.5, y=-0.5
        wall_dists_x_pos = 0.5 - c[:, 0]
        wall_dists_x_neg = 0.5 + c[:, 0]
        wall_dists_y_pos = 0.5 - c[:, 1]
        wall_dists_y_neg = 0.5 + c[:, 1]
        
        # Combine all distances for soft-min
        # We need the gradient, so we need to track which distance corresponds to which pair/wall
        
        # Soft-min value (for tracking)
        # To avoid overflow/underflow, use log-sum-exp
        # Let D be the array of distances
        # softmin = -1/beta * log(sum(exp(-beta * D)))
        
        # For gradient, we need weights w_k = exp(-beta * d_k) / sum(exp(-beta * d_k))
        # grad f = 1/beta * sum( w_k * grad d_k )
        
        # Let's compute the unnormalized weights
        # Pairwise
        exp_dists = np.exp(-beta * dists)
        # Set diagonal to 0
        np.fill_diagonal(exp_dists, 0)
        
        # Wall weights
        exp_wxp = np.exp(-beta * wall_dists_x_pos)
        exp_wxn = np.exp(-beta * wall_dists_x_neg)
        exp_wyp = np.exp(-beta * wall_dists_y_pos)
        exp_wyn = np.exp(-beta * wall_dists_y_neg)
        
        total_sum = np.sum(exp_dists) + np.sum(exp_wxp) + np.sum(exp_wxn) + np.sum(exp_wyp) + np.sum(exp_wyn)
        
        if total_sum == 0:
            break
            
        # Normalized weights
        w_dists = exp_dists / total_sum
        w_wxp = exp_wxp / total_sum
        w_wxn = exp_wxn / total_sum
        w_wyp = exp_wyp / total_sum
        w_wyn = exp_wyn / total_sum
        
        # Compute Gradient
        grad = np.zeros_like(c)
        
        # Pairwise contribution
        # grad_i = sum_j w_ij * (c_i - c_j) / d_ij
        # diff is c_i - c_j
        # dists is d_ij
        # We need to handle d_ij = 0 (should not happen if initialized well, but just in case)
        safe_dists = np.where(dists > 1e-8, dists, 1e-8)
        
        # term_ij = w_ij * (c_i - c_j) / d_ij
        # This is a (N, N, 2) tensor
        # w_dists is (N, N)
        # diff is (N, N, 2)
        # safe_dists is (N, N)
        
        # We want sum over j for each i
        # grad_pair = sum_j ( w_ij * diff_ij / d_ij )
        
        # Note: w_ij is symmetric, diff_ij = -diff_ji, d_ij = d_ji.
        
        # Vectorized:
        # factor = w_dists / safe_dists  (N, N)
        # grad_pair = sum_j factor_ij * diff_ij
        # diff is (N, N, 2). factor is (N, N).
        # We can do: np.sum(factor[..., np.newaxis] * diff, axis=1)
        
        factor = w_dists / safe_dists
        grad_pair = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
        
        # Wall contributions
        # x_pos: d = 0.5 - x. grad_d/dx = -1.
        # grad_x += sum_i w_wxp_i * (-1) * (-1) ?
        # f = -1/beta log(sum exp(-beta d))
        # df/dc = 1/beta * sum w_k * dc/dk
        # Wait, my previous derivation:
        # f = -1/beta log S
        # df/dc = -1/beta * (1/S) * dS/dc
        # dS/dc = sum exp(-beta d_k) * (-beta) * dd_k/dc
        # df/dc = 1/S * sum exp(-beta d_k) * dd_k/dc
        # df/dc = sum w_k * dd_k/dc
        # So, grad = sum w_k * grad d_k
        
        # Wall x_pos: d = 0.5 - x. grad d/dx = -1.
        # Contribution to grad_x: w_wxp * (-1)
        grad[:, 0] += -w_wxp
        # Wall x_neg: d = 0.5 + x. grad d/dx = 1.
        grad[:, 0] += w_wxn
        # Wall y_pos: d = 0.5 - y. grad d/dy = -1.
        grad[:, 1] += -w_wyp
        # Wall y_neg: d = 0.5 + y. grad d/dy = 1.
        grad[:, 1] += w_wyn
        
        # Total gradient
        grad += grad_pair
        
        # Step size
        # Adaptive step size
        # If beta is large, the gradient is concentrated on the minimum distance.
        # Step size should be small to avoid oscillations.
        # Let's use a fixed small step size scaled by 1/beta?
        # Or just a small constant.
        # The scale of distances is ~0.03.
        # Let's try eta = 0.001 / beta * 10?
        # Actually, the gradient magnitude is roughly 1 (since weights sum to 1, and d/dc is ~1).
        # So a step of 0.001 is reasonable.
        # As beta increases, the "effective" stiffness increases, so we might need smaller steps.
        eta = 0.005 / (1 + beta/100)
        
        c += eta * grad
        
        # Clip to square
        c = np.clip(c, -0.5, 0.5)
        
        # Calculate score
        # min dist
        min_pair = np.min(dists)
        min_wall = np.min([np.min(wall_dists_x_pos), np.min(wall_dists_x_neg), 
                           np.min(wall_dists_y_pos), np.min(wall_dists_y_neg)])
        r = min(min_pair, min_wall) / 2.0
        
        if r > best_score:
            best_score = r
            best_c = c.copy()
            
        # Early stopping if no improvement for a long time?
        # Keep it simple: run until time limit.
        
    # Final polish:
    # Do a few steps with very small eta to settle
    if time.time() - start_time < deadline - 5:
        for _ in range(100):
            beta = beta_end
            # Recalculate grad for best_c
            c = best_c.copy()
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            
            wall_dists_x_pos = 0.5 - c[:, 0]
            wall_dists_x_neg = 0.5 + c[:, 0]
            wall_dists_y_pos = 0.5 - c[:, 1]
            wall_dists_y_neg = 0.5 + c[:, 1]
            
            exp_dists = np.exp(-beta * dists)
            np.fill_diagonal(exp_dists, 0)
            
            exp_wxp = np.exp(-beta * wall_dists_x_pos)
            exp_wxn = np.exp(-beta * wall_dists_x_neg)
            exp_wyp = np.exp(-beta * wall_dists_y_pos)
            exp_wyn = np.exp(-beta * wall_dists_y_neg)
            
            total_sum = np.sum(exp_dists) + np.sum(exp_wxp) + np.sum(exp_wxn) + np.sum(exp_wyp) + np.sum(exp_wyn)
            if total_sum == 0: break
            
            w_dists = exp_dists / total_sum
            w_wxp = exp_wxp / total_sum
            w_wxn = exp_wxn / total_sum
            w_wyp = exp_wyp / total_sum
            w_wyn = exp_wyn / total_sum
            
            safe_dists = np.where(dists > 1e-8, dists, 1e-8)
            factor = w_dists / safe_dists
            grad_pair = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
            
            grad = np.zeros_like(c)
            grad[:, 0] += -w_wxp
            grad[:, 0] += w_wxn
            grad[:, 1] += -w_wyp
            grad[:, 1] += w_wyn
            grad += grad_pair
            
            eta = 0.0005
            c += eta * grad
            c = np.clip(c, -0.5, 0.5)
            
            min_pair = np.min(dists)
            min_wall = np.min([np.min(wall_dists_x_pos), np.min(wall_dists_x_neg), 
                           np.min(wall_dists_y_pos), np.min(wall_dists_y_neg)])
            r = min(min_pair, min_wall) / 2.0
            
            if r > best_score:
                best_score = r
                best_c = c.copy()
                
            if time.time() - start_time > deadline - 1:
                break
                
    return best_c
