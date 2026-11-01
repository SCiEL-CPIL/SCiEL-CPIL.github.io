"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_score(centers):
    """Compute the packing radius: min(min_pair_dist/2, min_wall_dist)."""
    wall = 0.5 - np.abs(centers)
    min_wall = np.min(wall)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, np.inf)
    min_pair = np.sqrt(np.min(d2))
    return min(min_pair / 2.0, min_wall)


def _sa_run(centers, rng, time_budget, seed=42):
    """
    Efficient simulated annealing with incremental score tracking.
    Returns (best_centers, best_score).
    """
    start = time.time()
    centers = centers.copy()
    
    # Precompute wall distances per circle (min of x-wall and y-wall)
    wall_per_circle = np.min(0.5 - np.abs(centers), axis=1)  # shape (N,)
    
    # Precompute full distance matrix
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, np.inf)
    
    min_pair_sq = np.min(d2)
    min_wall = np.min(wall_per_circle)
    cur_score = min(np.sqrt(min_pair_sq) / 2.0, min_wall)
    
    best_centers = centers.copy()
    best_score = cur_score
    
    # SA parameters
    T0 = 0.02
    Tend = 1e-10
    # We'll run for a fixed number of iterations and check time
    max_iters = 500000
    
    # Adaptive step size
    step = 0.025
    base_step = 0.025
    no_improve = 0
    patience = 3000
    
    # Track the minimum pair and which pair it is, for incremental updates
    # We'll maintain d2 and track min_pair_sq
    # For wall, we track wall_per_circle
    
    for it in range(max_iters):
        if it % 1000 == 0:
            if time.time() - start > time_budget:
                break
        
        # Temperature schedule: exponential cooling
        progress = it / max_iters
        T = T0 * (Tend / T0) ** progress
        
        # Choose circle to move
        i = rng.integers(N)
        old_pos = centers[i].copy()
        old_wall_i = wall_per_circle[i]
        
        # Generate perturbation
        scale = step * (1.0 + 0.5 * rng.random())
        new_pos = old_pos + rng.normal(0, scale, 2)
        new_pos = np.clip(new_pos, -0.4999, 0.4999)
        
        if np.allclose(new_pos, old_pos):
            continue
        
        # Compute new distances from i to all others
        new_d2_i = np.sum((centers - new_pos) ** 2, axis=1)
        new_d2_i[i] = np.inf
        
        # New wall for circle i
        new_wall_i = np.min(0.5 - np.abs(new_pos))
        
        # Compute new global min pair sq
        # The old min_pair_sq might involve i or not
        # new_min_pair_sq = min(min of d2 excluding row/col i, new_d2_i)
        # To avoid O(N^2) each time, we track:
        # - If the current min pair involves i, we need to find the next min
        # Simpler: just compute min over all pairs not involving i, then min with new_d2_i
        
        # Find min pair not involving i:
        # We can do this by masking row and col i
        # But that's O(N^2). Instead, let's keep a sorted list or use a trick.
        
        # For efficiency, let's compute the new min_pair_sq as:
        # min(min_pair_sq_not_involving_i, min(new_d2_i))
        # where min_pair_sq_not_involving_i is the min of d2 with row/col i removed
        
        # Quick approach: if the current min pair doesn't involve i, then
        # min_pair_sq_not_involving_i = min_pair_sq
        # Otherwise, we need to find the second minimum
        
        # Let's track which pair gives the min
        # Find argmin of d2
        min_idx = np.unravel_index(np.argmin(d2), d2.shape)
        min_pair_involved_i = (min_idx[0] == i or min_idx[1] == i)
        
        if not min_pair_involved_i:
            min_pair_sq_excl = min_pair_sq
        else:
            # Find the second minimum pair not involving i
            # Mask out row and col i
            mask = np.ones((N, N), dtype=bool)
            mask[i, :] = False
            mask[:, i] = False
            d2_masked = np.where(mask, d2, np.inf)
            min_pair_sq_excl = np.min(d2_masked)
        
        new_min_pair_sq = min(min_pair_sq_excl, np.min(new_d2_i))
        
        # New min wall
        if old_wall_i >= min_wall:
            # The min wall might not be from circle i
            # Find min wall excluding i
            if N > 1:
                wall_excl = np.delete(wall_per_circle, i)
                min_wall_excl = np.min(wall_excl)
            else:
                min_wall_excl = np.inf
            new_min_wall = min(new_wall_i, min_wall_excl)
        else:
            # old_wall_i was the min, so new min is min(new_wall_i, second min)
            if N > 1:
                wall_excl = np.delete(wall_per_circle, i)
                min_wall_excl = np.min(wall_excl)
            else:
                min_wall_excl = np.inf
            new_min_wall = min(new_wall_i, min_wall_excl)
        
        new_score = min(np.sqrt(new_min_pair_sq) / 2.0, new_min_wall)
        
        # Acceptance criterion
        delta = new_score - cur_score
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-15)):
            # Accept the move
            centers[i] = new_pos
            wall_per_circle[i] = new_wall_i
            
            # Update d2
            diff_i = centers - new_pos
            d2[i, :] = np.sum(diff_i ** 2, axis=1)
            d2[:, i] = d2[i, :]
            d2[i, i] = np.inf
            
            cur_score = new_score
            no_improve = 0
            
            if cur_score > best_score:
                best_score = cur_score
                best_centers = centers.copy()
        else:
            no_improve += 1
        
        # Adaptive step size
        if no_improve > patience:
            step *= 0.7
            no_improve = 0
        elif it % 5000 == 0 and it > 0:
            # Occasionally reset step to explore
            if rng.random() < 0.3:
                step = base_step * (0.8 + 0.4 * rng.random())
                no_improve = 0
    
    # Final local refinement
    remaining = time_budget - (time.time() - start)
    if remaining > 5.0:
        centers = best_centers.copy()
        refine_start = time.time()
        refine_budget = min(remaining - 1.0, 20.0)
        
        # Greedy hill climbing with decreasing step
        refine_step = 0.008
        cur_score = _compute_score(centers)
        
        while time.time() - refine_start < refine_budget:
            improved = False
            for _ in range(1000):
                i = rng.integers(N)
                old_pos = centers[i].copy()
                new_pos = old_pos + rng.normal(0, refine_step, 2)
                new_pos = np.clip(new_pos, -0.4999, 0.4999)
                centers[i] = new_pos
                new_score = _compute_score(centers)
                
                if new_score > cur_score:
                    cur_score = new_score
                    improved = True
                    if cur_score > best_score:
                        best_score = cur_score
                        best_centers = centers.copy()
                else:
                    centers[i] = old_pos
            
            if not improved:
                refine_step *= 0.5
                if refine_step < 1e-7:
                    break
    
    return best_centers, best_score


def construct():
    """
    Multi-start simulated annealing to maximize the packing radius.
    """
    rng = np.random.default_rng(12345)
    
    start = time.time()
    max_time = 110.0
    
    best_centers = None
    best_score = -np.inf
    
    # Generate diverse initializations
    inits = []
    
    # 1. Hexagonal packing (best known structure)
    hex_sp = 0.19
    hex_h = hex_sp * np.sqrt(3) / 2
    hex_pts = []
    for row in range(12):
        offset = 0 if row % 2 == 0 else hex_sp / 2
        for col in range(12):
            x = -0.48 + offset + col * hex_sp
            y = -0.48 + row * hex_h
            if -0.4999 <= x <= 0.4999 and -0.4999 <= y <= 0.4999:
                hex_pts.append([x, y])
    hex_pts = np.array(hex_pts[:N])
    if len(hex_pts) == N:
        hex_pts = hex_pts + rng.normal(0, 0.002, (N, 2))
        hex_pts = np.clip(hex_pts, -0.4999, 0.4999)
        inits.append(hex_pts)
    
    # 2. Grid packings with different configurations
    grid_configs = [(7, 8), (8, 7), (5, 10), (10, 5), (6, 9), (9, 6)]
    for rows, cols in grid_configs:
        x_ticks = np.linspace(-0.47, 0.47, cols)
        y_ticks = np.linspace(-0.47, 0.47, rows)
        pts = []
        for r in range(rows):
            for c in range(cols):
                if len(pts) < N:
                    pts.append([x_ticks[c], y_ticks[r]])
        if len(pts) == N:
            pts = np.array(pts)
            pts = pts + rng.normal(0, 0.002, (N, 2))
            pts = np.clip(pts, -0.4999, 0.4999)
            inits.append(pts)
    
    # 3. Random initialization
    inits.append(rng.uniform(-0.45, 0.45, (N, 2)))
    
    # 4. Another hexagonal with slightly different spacing
    hex_sp2 = 0.20
    hex_h2 = hex_sp2 * np.sqrt(3) / 2
    hex_pts2 = []
    for row in range(12):
        offset = 0 if row % 2 == 0 else hex_sp2 / 2
        for col in range(12):
            x = -0.47 + offset + col * hex_sp2
            y = -0.47 + row * hex_h2
            if -0.4999 <= x <= 0.4999 and -0.4999 <= y <= 0.4999:
                hex_pts2.append([x, y])
    if len(hex_pts2) >= N:
        hex_pts2 = np.array(hex_pts2[:N])
        hex_pts2 = hex_pts2 + rng.normal(0, 0.002, (N, 2))
        hex_pts2 = np.clip(hex_pts2, -0.4999, 0.4999)
        inits.append(hex_pts2)
    
    # Run SA from each initialization
    for idx, init in enumerate(inits):
        remaining = max_time - (time.time() - start)
        if remaining < 8.0:
            break
        
        # Allocate time: more time for earlier starts, but ensure at least 8s
        num_left = len(inits) - idx
        time_budget = min(remaining - 1.0, max(8.0, (remaining - 1.0) / num_left))
        
        c, s = _sa_run(init, rng, time_budget, seed=12345 + idx)
        
        if s > best_score:
            best_score = s
            best_centers = c
    
    # Fallback
    if best_centers is None:
        x_ticks = np.linspace(-0.47, 0.47, 8)
        y_ticks = np.linspace(-0.47, 0.47, 7)
        best_centers = np.zeros((N, 2))
        idx = 0
        for r in range(7):
            for c in range(8):
                if idx < N:
                    best_centers[idx] = [x_ticks[c], y_ticks[r]]
                    idx += 1
                else:
                    break
            if idx >= N:
                break
    
    return best_centers
# EVOLVE-BLOCK-END
