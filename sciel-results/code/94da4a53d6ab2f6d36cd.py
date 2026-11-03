# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    time_limit = 145.0 # Safety margin

    def calc_radius(c):
        """
        Calculate the minimum radius for centers c.
        Returns (radius, limiting_pairs, limiting_walls)
        limiting_pairs: indices (i, j) where distance is minimal
        limiting_walls: indices (i, wall) where wall distance is minimal
        """
        # Pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        
        # Find min pairwise distance
        min_pair_idx = np.unravel_index(np.argmin(d), d.shape)
        min_pair_dist = d[min_pair_idx]
        
        # Wall distances
        # We need the global minimum wall distance
        # For each point i, min wall dist is min(0.5 - max(c_i), min(c_i) + 0.5) ? No.
        # min(0.5 - x, x + 0.5) is the distance to the nearest wall in that dimension.
        # Overall wall dist for point i is min over dimensions of that value.
        
        # Vectorized:
        # dist_to_walls_x = np.minimum(0.5 - c[:,0], c[:,0] + 0.5)
        # etc.
        # But we need to know WHICH wall for the gradient/push.
        
        # Let's compute the 6 wall distances for each point.
        d_x_pos = 0.5 - c[:, 0]
        d_x_neg = c[:, 0] + 0.5
        d_y_pos = 0.5 - c[:, 1]
        d_y_neg = c[:, 1] + 0.5
        d_z_pos = 0.5 - c[:, 2]
        d_z_neg = c[:, 2] + 0.5
        
        wall_dists = np.stack([d_x_pos, d_x_neg, d_y_pos, d_y_neg, d_z_pos, d_z_neg], axis=1)
        # wall_dists shape (N, 6)
        
        min_wall_idx = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
        min_wall_dist = wall_dists[min_wall_idx]
        
        # Determine global minimum
        if min_pair_dist / 2.0 < min_wall_dist:
            radius = min_pair_dist / 2.0
            # Collect all pairs with distance close to min
            # Use a threshold to find "active" constraints for repulsion
            threshold = min_pair_dist * 1.01 + 1e-9
            active_pairs = np.argwhere(d <= threshold)
            # Remove self pairs (diagonal is inf, so they won't be selected unless threshold is huge)
            # Ensure i < j to avoid duplicates if needed, but for repulsion direction sign matters
            # We just need the vectors.
            return radius, active_pairs, None
        else:
            radius = min_wall_dist
            # Collect all wall constraints close to min
            threshold = min_wall_dist * 1.01 + 1e-9
            active_walls = np.argwhere(wall_dists <= threshold)
            # active_walls contains (point_idx, wall_idx)
            return radius, None, active_walls

    def local_search(c, max_iters=200, step_size=0.01, time_budget=None):
        """
        Perform local search by pushing spheres apart from limiting constraints.
        """
        c = c.copy()
        current_r, pairs, walls = calc_radius(c)
        
        # Adaptive step size
        step = step_size
        
        for i in range(max_iters):
            if time_budget is not None and time.time() - start_time > time_budget:
                break
                
            # Recalculate constraints
            current_r, pairs, walls = calc_radius(c)
            
            # Compute forces
            forces = np.zeros_like(c)
            
            if pairs is not None and len(pairs) > 0:
                for p in pairs:
                    i, j = p[0], p[1]
                    if i == j: continue
                    # If i < j, vector is c_i - c_j. Push i away from j (add), push j away from i (sub)
                    # But if we iterate all i,j, we double count.
                    # Let's just use the unique pairs from argwhere? argwhere gives all i,j.
                    # To be symmetric, we can just sum vectors.
                    diff = c[i] - c[j]
                    dist = np.linalg.norm(diff)
                    if dist > 1e-9:
                        unit_vec = diff / dist
                        # Push i in direction of unit_vec, j in direction of -unit_vec
                        forces[i] += unit_vec
                        forces[j] -= unit_vec
            elif walls is not None and len(walls) > 0:
                for w in walls:
                    i, wall_idx = w[0], w[1]
                    # wall_idx: 0:x+, 1:x-, 2:y+, 3:y-, 4:z+, 5:z-
                    # Push away from wall
                    if wall_idx == 0: forces[i, 0] -= 1.0 # Wall at 0.5, push negative x
                    elif wall_idx == 1: forces[i, 0] += 1.0 # Wall at -0.5, push positive x
                    elif wall_idx == 2: forces[i, 1] -= 1.0
                    elif wall_idx == 3: forces[i, 1] += 1.0
                    elif wall_idx == 4: forces[i, 2] -= 1.0
                    elif wall_idx == 5: forces[i, 2] += 1.0
            
            # Normalize forces to avoid huge jumps
            force_norms = np.linalg.norm(forces, axis=1)
            # Only update if there is a force
            mask = force_norms > 1e-9
            if np.any(mask):
                forces[mask] /= force_norms[mask, np.newaxis]
                
                # Update positions
                c += step * forces
                # Clip to bounds (soft clip, actually just ensure inside)
                c = np.clip(c, -0.5 + 1e-6, 0.5 - 1e-6)
                
                # Check improvement
                new_r, _, _ = calc_radius(c)
                if new_r > current_r:
                    current_r = new_r
                    # Optionally increase step slightly? Or keep constant.
                    # step = min(step * 1.05, 0.05)
                else:
                    # Revert
                    c -= step * forces
                    c = np.clip(c, -0.5 + 1e-6, 0.5 - 1e-6)
                    # Decrease step
                    step *= 0.8
                    if step < 1e-6:
                        break
            
        return c, current_r

    def greedy_init():
        """
        Initialize by placing spheres one by one to maximize min distance.
        """
        centers = np.zeros((N, 3))
        centers[0] = [0, 0, 0]
        
        for i in range(1, N):
            best_pos = None
            best_r = -1
            num_trials = 1000
            candidates = np.random.uniform(-0.4, 0.4, (num_trials, 3))
            
            # Calculate distances to existing centers (i, 3) vs (num_trials, 3)
            # diff: (num_trials, i, 3)
            diff = candidates[:, np.newaxis, :] - centers[:i][np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2) # (num_trials, i)
            min_dists_to_centers = np.min(dists, axis=1) # (num_trials,)
            
            # Wall distances
            # min(0.5 - c, c + 0.5) for each coord, then min over coords
            min_dists_to_walls = np.min(np.minimum(0.5 - candidates, candidates + 0.5), axis=1)
            
            # Radius for each candidate
            radii = np.minimum(min_dists_to_centers / 2.0, min_dists_to_walls)
            
            best_idx = np.argmax(radii)
            best_r = radii[best_idx]
            best_pos = candidates[best_idx].copy()
            
            centers[i] = best_pos
        
        return centers

    def grid_init():
        """
        Generate a grid initialization.
        """
        m = 4
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N].copy()

    # Generate Candidates
    np.random.seed(42)
    
    candidates = []
    
    # 1. Greedy
    candidates.append(greedy_init())
    
    # 2. Grid
    candidates.append(grid_init())
    
    # 3. Random
    candidates.append(np.random.uniform(-0.4, 0.4, (N, 3)))
    
    # 4. Perturbed Grid
    candidates.append(grid_init() + np.random.uniform(-0.05, 0.05, (N, 3)))
    
    best_c = None
    best_r = -1
    
    # Run Local Search on each candidate
    for i, c_init in enumerate(candidates):
        # Check time
        if time.time() - start_time > time_limit - 10:
            break
            
        # Run local search
        # Give each candidate a time budget
        time_budget = time_limit - 10 - i * 5 # Decreasing budget? No, just check global.
        
        c_opt, r_opt = local_search(c_init, max_iters=500, step_size=0.02, time_budget=time_limit - 5)
        
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt.copy()
            
            # If we find a good one, try to refine it with smaller steps
            if time.time() - start_time < time_limit - 10:
                c_refine, r_refine = local_search(c_opt, max_iters=500, step_size=0.005, time_budget=time_limit - 5)
                if r_refine > best_r:
                    best_r = r_refine
                    best_c = c_refine.copy()

    # If no valid result found (should not happen), return grid
    if best_c is None:
        best_c = grid_init()

    return best_c

# EVOLVE-BLOCK-END
