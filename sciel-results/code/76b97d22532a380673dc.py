"""Optimized packing of 200 equal circles in a unit square."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    # 1. Initialization: Jittered Grid
    m = 15
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    x = grid[:N].copy()
    
    np.random.seed(42)
    x += np.random.uniform(-0.01, 0.01, size=x.shape)
    x = np.clip(x, -0.49, 0.49)
    
    # Precompute distance matrix
    def calc_dist_matrix(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        return dist

    D = calc_dist_matrix(x)
    
    def calc_min_radius_from_D(D, x):
        min_pair = np.min(D)
        d_left = x[:, 0] + 0.5
        d_right = 0.5 - x[:, 0]
        d_bottom = x[:, 1] + 0.5
        d_top = 0.5 - x[:, 1]
        min_bound = np.min(np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top)))
        return min(min_pair, min_bound)

    r = calc_min_radius_from_D(D, x)
    
    max_iters = 50000
    start_time = time.time()
    candidates_per_move = 20
    
    for _ in range(max_iters):
        if time.time() - start_time > 115:
            break
            
        # Pick a random circle to move
        idx = np.random.randint(N)
        
        # Calculate R_other: min distance among all pairs not involving idx
        # Copy D and set row/col to inf
        # To avoid copy, we can just find min of D with D[idx,:] and D[:,idx] ignored
        # A simple way: flatten D, mask out idx
        # Or just: min( min(D[:, :idx]), min(D[:, idx+1:]), min(D[:idx, :]), min(D[idx+1:, :]) )
        # This is O(N^2) if done naively.
        # Better: maintain a sorted list? No.
        # Since N=200, O(N^2) is 40k ops.
        # If we do this every iteration, 50k * 40k = 2B ops. Too slow.
        
        # Alternative: Just pick the circle with the smallest clearance.
        # We need min_pair_per_circle and min_bound_per_circle.
        min_pair_per_circle = np.min(D, axis=1)
        d_left = x[:, 0] + 0.5
        d_right = 0.5 - x[:, 0]
        d_bottom = x[:, 1] + 0.5
        d_top = 0.5 - x[:, 1]
        min_bound_per_circle = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
        min_clearance = np.minimum(min_pair_per_circle, min_bound_per_circle)
        
        idx = np.argmin(min_clearance)
        
        # Now calculate R_other for this specific idx
        # We need the min of D excluding row and col idx
        # We can create a view or mask
        # D_masked = D.copy(); D_masked[idx, :] = np.inf; D_masked[:, idx] = np.inf
        # R_other = np.min(D_masked)
        # This is O(N^2) per iteration.
        
        # Can we avoid O(N^2) for R_other?
        # If we maintain the global min and second min...
        # Let's just accept O(N^2) for R_other but reduce iterations.
        # 10,000 iterations * 40,000 ops = 400M ops. Feasible.
        
        D_masked = D.copy()
        D_masked[idx, :] = np.inf
        D_masked[:, idx] = np.inf
        R_other = np.min(D_masked)
        
        best_new_r = r
        best_new_pos = None
        
        # Generate candidates
        # 10 global, 10 local
        for _ in range(10):
            new_pos = np.random.uniform(-0.49, 0.49, size=2)
            # Eval
            diffs = x - new_pos
            diffs[idx] = [10, 10]
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            d_left = new_pos[0] + 0.5
            d_right = 0.5 - new_pos[0]
            d_bottom = new_pos[1] + 0.5
            d_top = 0.5 - new_pos[1]
            d_bound = min(d_left, d_right, d_bottom, d_top)
            min_cand = min(np.min(dists), d_bound)
            new_r = min(min_cand, R_other)
            
            if new_r > best_new_r:
                best_new_r = new_r
                best_new_pos = new_pos
                
        current_pos = x[idx]
        for _ in range(10):
            new_pos = current_pos + np.random.uniform(-0.05, 0.05, size=2)
            new_pos = np.clip(new_pos, -0.49, 0.49)
            diffs = x - new_pos
            diffs[idx] = [10, 10]
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            d_left = new_pos[0] + 0.5
            d_right = 0.5 - new_pos[0]
            d_bottom = new_pos[1] + 0.5
            d_top = 0.5 - new_pos[1]
            d_bound = min(d_left, d_right, d_bottom, d_top)
            min_cand = min(np.min(dists), d_bound)
            new_r = min(min_cand, R_other)
            
            if new_r > best_new_r:
                best_new_r = new_r
                best_new_pos = new_pos

        if best_new_pos is not None and best_new_r > r:
            x[idx] = best_new_pos
            # Update D
            diffs = x - best_new_pos
            diffs[idx] = [0, 0] # Will be set to inf in D anyway
            new_dists = np.sqrt(np.sum(diffs**2, axis=1))
            D[idx, :] = new_dists
            D[:, idx] = new_dists
            D[idx, idx] = np.inf
            r = best_new_r
        else:
            # Stuck, add noise
            if np.random.rand() < 0.1:
                x[idx] += np.random.uniform(-0.02, 0.02, size=2)
                x[idx] = np.clip(x[idx], -0.49, 0.49)
                # Update D
                diffs = x - x[idx]
                new_dists = np.sqrt(np.sum(diffs**2, axis=1))
                D[idx, :] = new_dists
                D[:, idx] = new_dists
                D[idx, idx] = np.inf
                # Recalculate r
                min_pair = np.min(D)
                d_left = x[:, 0] + 0.5
                d_right = 0.5 - x[:, 0]
                d_bottom = x[:, 1] + 0.5
                d_top = 0.5 - x[:, 1]
                min_bound = np.min(np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top)))
                r = min(min_pair, min_bound)

    return x
# EVOLVE-BLOCK-END
