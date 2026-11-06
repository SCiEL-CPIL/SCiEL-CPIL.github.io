import numpy as np
import time

N = 200

def _compute_score(centers):
    """Compute the radius for a given set of centers."""
    if len(centers) == 0 or not np.all(np.isfinite(centers)):
        return 0.0
    
    # Check bounds
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
        
    # Distance to walls
    # min(0.5 - x, x + 0.5) for each coordinate, then min over all points and coords
    wall_dist = np.min(np.minimum(0.5 - centers, centers + 0.5))
    
    if wall_dist <= 0:
        return 0.0

    # Distance to neighbors
    # Vectorized pairwise distance
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_dist_sq = np.min(dist_sq)
    min_dist = np.sqrt(min_dist_sq)
    
    r = min(wall_dist, min_dist / 2.0)
    return r

def _push_apart(centers, max_iters=5000, learning_rate=0.1, time_limit=None, start_time=None):
    """
    Iterative local search using repulsion forces.
    Each step:
    1. Compute repulsive forces between all pairs (inversely proportional to distance squared).
    2. Move centers slightly in the direction of the net force.
    3. Project centers back into the square [-0.5, 0.5]^2.
    """
    centers = centers.copy()
    n = len(centers)
    
    if time_limit is None:
        time_limit = 140.0
    if start_time is None:
        start_time = time.time()

    # Adaptive learning rate
    lr = learning_rate
    best_local_score = _compute_score(centers)
    best_local_centers = centers.copy()
    
    # We use a simple heuristic: if score doesn't improve for a while, reduce lr
    stall_count = 0
    last_improvement_score = best_local_score
    
    for i in range(max_iters):
        # Time check
        if time.time() - start_time > time_limit:
            break
            
        # 1. Compute forces
        # diff[i, j] = centers[i] - centers[j]
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, 1.0) # Avoid division by zero
        
        # Force magnitude: 1 / dist^2 (soft repulsion)
        # We want to move away from neighbors.
        # Force on i from j is proportional to (centers[i] - centers[j]) / dist^2
        # Summing over j: F_i = sum_j (centers[i] - centers[j]) / dist_ij^2
        
        # To be efficient, we can compute:
        # M[i, j] = 1 / dist_sq[i, j]
        M = 1.0 / dist_sq
        np.fill_diagonal(M, 0.0)
        
        # Sum of M * centers[j] for each i
        # S_i = sum_j M[i, j] * centers[j]
        S = M @ centers
        
        # Sum of M for each i
        # RowSum_i = sum_j M[i, j]
        RowSum = M.sum(axis=1)
        
        # Net force direction (unnormalized):
        # F_i = sum_j M[i, j] * (centers[i] - centers[j])
        # F_i = centers[i] * RowSum_i - S_i
        forces = centers * RowSum[:, np.newaxis] - S
        
        # 2. Update positions
        centers += lr * forces
        
        # 3. Project to square
        centers = np.clip(centers, -0.5, 0.5)
        
        # 4. Check for improvement
        # We don't want to compute full score every iteration (O(N^2) is fine for N=200, but let's be careful)
        # Actually, for N=200, O(N^2) is 40,000 ops, which is very fast.
        # Let's compute score every 10 iterations to save overhead, or every iteration if it's fast enough.
        if i % 5 == 0 or i == max_iters - 1:
            current_score = _compute_score(centers)
            if current_score > best_local_score:
                best_local_score = current_score
                best_local_centers = centers.copy()
                stall_count = 0
                # Slightly increase lr if we are improving rapidly? Or keep it.
                # Let's keep lr constant or decay slowly.
            else:
                stall_count += 1
                if stall_count > 50: # 50 * 5 = 250 iters without improvement
                    lr *= 0.9
                    if lr < 1e-5:
                        break
                    stall_count = 0
                    
    return best_local_centers, best_local_score

def _generate_hex_grid(N):
    """Generate a hexagonal grid of points."""
    # Estimate d based on area
    # Area per circle in hex packing ~ sqrt(3)/2 * d^2
    # Total Area = 1.0
    # N * sqrt(3)/2 * d^2 = 1.0 => d = sqrt(2 / (N * sqrt(3)))
    d = np.sqrt(2.0 / (N * np.sqrt(3.0))) * 0.95 # Slightly smaller to ensure fit
    
    points = []
    row_spacing = d * np.sqrt(3.0) / 2.0
    num_rows = int(np.ceil(1.0 / row_spacing)) + 2
    
    for row in range(num_rows):
        y = -0.5 + row * row_spacing
        if y > 0.5:
            break
        offset = d / 2.0 if row % 2 else 0.0
        num_cols = int(np.ceil(1.0 / d)) + 2
        for col in range(num_cols):
            x = -0.5 + col * d + offset
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                points.append([x, y])
                
    points = np.array(points)
    if len(points) < N:
        return None
    return points[:N]

def _generate_square_grid(N):
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]

def construct():
    start_time = time.time()
    time_limit = 145.0 # Safety margin
    
    best_centers = None
    best_score = -1.0
    
    # 1. Generate diverse initializations
    inits = []
    
    # Hex Grid
    hex_grid = _generate_hex_grid(N)
    if hex_grid is not None:
        inits.append(hex_grid)
        
    # Square Grid
    square_grid = _generate_square_grid(N)
    inits.append(square_grid)
    
    # Jittered Hex
    if hex_grid is not None:
        np.random.seed(42)
        jittered_hex = hex_grid + np.random.normal(0, 0.01, hex_grid.shape)
        jittered_hex = np.clip(jittered_hex, -0.49, 0.49)
        inits.append(jittered_hex)
        
    # Random Initializations
    # We will generate these on the fly if time permits, but let's add a few fixed ones
    for seed in [1, 2, 3]:
        np.random.seed(seed)
        rand_centers = np.random.uniform(-0.49, 0.49, (N, 2))
        inits.append(rand_centers)
        
    # Sort initializations by their initial score (best first) to prioritize good starts
    scored_inits = []
    for init in inits:
        score = _compute_score(init)
        scored_inits.append((score, init))
    scored_inits.sort(key=lambda x: x[0], reverse=True)
    
    # 2. Run local search on each initialization
    # The push_apart method is much faster than L-BFGS-B, so we can try all of them
    for score, init_centers in scored_inits:
        if time.time() - start_time > time_limit:
            break
            
        # Run local search
        # We allow more iterations for better starts? Or just run until time/stall.
        # Let's give each one a fair chunk of time, or just run until stall/time.
        # Since push_apart is fast, we can run many iterations.
        # Let's cap iterations per run to avoid one run eating all time if it doesn't stall quickly.
        # 10,000 iterations is usually enough for convergence in this method.
        centers, local_score = _push_apart(init_centers, max_iters=10000, learning_rate=0.05, time_limit=time_limit, start_time=start_time)
        
        if local_score > best_score:
            best_score = local_score
            best_centers = centers
            
    # 3. If we still have time, do some random restarts from the best found so far with perturbation
    if best_centers is not None and time.time() - start_time < time_limit - 10:
        for _ in range(5): # 5 more restarts
            if time.time() - start_time > time_limit:
                break
            np.random.seed(int(time.time() * 1000) % 10000)
            # Perturb best solution
            perturbed = best_centers + np.random.normal(0, 0.02, (N, 2))
            perturbed = np.clip(perturbed, -0.49, 0.49)
            
            centers, local_score = _push_apart(perturbed, max_iters=5000, learning_rate=0.02, time_limit=time_limit, start_time=start_time)
            
            if local_score > best_score:
                best_score = local_score
                best_centers = centers
                
    # Fallback if something went wrong
    if best_centers is None:
        best_centers = inits[0]
        
    return best_centers
