"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time


def compute_score(centers):
    """Compute the packing score: minimum radius."""
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5 - 1e-12) or np.any(centers > 0.5 + 1e-12):
        return 0.0
    wall_dist = np.min(0.5 - np.abs(centers), axis=1)
    min_wall = np.min(wall_dist)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    return min(min_wall, min_pair / 2.0)


def init_grid(perturb=0.01, seed=0):
    rng = np.random.RandomState(seed)
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])[:N]
    grid = grid + rng.uniform(-perturb, perturb, size=grid.shape)
    return np.clip(grid, -0.45, 0.45)


def init_random(seed=0):
    rng = np.random.RandomState(seed)
    return rng.uniform(-0.45, 0.45, size=(N, 3))


def pairwise_distances(centers):
    """Compute all pairwise distances."""
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))


def relax(centers, max_iters=500):
    """Relax using incremental distance tracking."""
    centers = centers.copy()
    dist = pairwise_distances(centers)
    np.fill_diagonal(dist, np.inf)
    
    wall_dist = 0.5 - np.abs(centers)
    min_wall = np.min(np.min(wall_dist, axis=1))
    min_pair = np.min(dist)
    best_score = min(min_wall, min_pair / 2.0)
    best_centers = centers.copy()
    best_dist = dist.copy()
    best_wall_dist = wall_dist.copy()
    
    for it in range(max_iters):
        improved = False
        for i in range(N):
            # Get current min distance involving sphere i
            row_min = np.min(dist[i])
            wall_min_i = np.min(wall_dist[i])
            
            # Find nearest neighbor
            nearest_j = np.argmin(dist[i])
            nearest_dist = dist[i, nearest_j]
            
            # Find nearest wall axis
            nearest_wall_axis = np.argmin(wall_dist[i])
            nearest_wall_dist = wall_dist[i, nearest_wall_axis]
            
            # Determine which constraint is limiting
            if nearest_dist / 2.0 <= nearest_wall_dist:
                direction = (centers[i] - centers[nearest_j]) / (nearest_dist + 1e-15)
                limit = nearest_dist / 2.0
            else:
                direction = np.zeros(3)
                direction[nearest_wall_axis] = -np.sign(centers[i, nearest_wall_axis])
                if direction[nearest_wall_axis] == 0:
                    direction[nearest_wall_axis] = 1.0
                limit = nearest_wall_dist
            
            # Try moving
            for step in [0.003, 0.001]:
                new_center = centers[i] + step * direction
                if np.any(new_center < -0.4999) or np.any(new_center > 0.4999):
                    continue
                
                # Update distances involving sphere i
                new_dist_i = np.sqrt(np.sum((centers - new_center) ** 2, axis=1))
                new_dist_i[i] = np.inf
                new_wall_i = 0.5 - np.abs(new_center)
                
                # Check if this improves the global minimum
                new_global_min_pair = min(np.min(dist), np.min(new_dist_i))
                new_global_min_wall = min(np.min(wall_dist), np.min(new_wall_i))
                new_score = min(new_global_min_wall, new_global_min_pair / 2.0)
                
                if new_score > best_score + 1e-12:
                    # Update state
                    dist[i] = new_dist_i
                    dist[new_dist_i != np.inf, i] = new_dist_i[new_dist_i != np.inf]
                    # Actually need to update symmetrically
                    for j in range(N):
                        if j != i:
                            dist[j, i] = new_dist_i[j]
                            dist[i, j] = new_dist_i[j]
                    wall_dist[i] = new_wall_i
                    centers[i] = new_center
                    best_score = new_score
                    best_centers = centers.copy()
                    best_dist = dist.copy()
                    best_wall_dist = wall_dist.copy()
                    improved = True
                    break
                elif new_score > min(np.min(wall_dist), np.min(dist) / 2.0) + 1e-12:
                    # Accept even if not improving best, if it improves current
                    dist[i] = new_dist_i
                    for j in range(N):
                        if j != i:
                            dist[j, i] = new_dist_i[j]
                            dist[i, j] = new_dist_i[j]
                    wall_dist[i] = new_wall_i
                    centers[i] = new_center
                    improved = True
                    break
        
        if not improved:
            break
    
    return best_centers, best_score


def construct():
    start_time = time.time()
    deadline = 135.0
    
    best_centers = None
    best_score = 0.0
    
    attempts = []
    for seed in range(3):
        attempts.append(init_grid(perturb=0.01, seed=seed))
    for seed in range(3):
        attempts.append(init_random(seed=seed))
    
    for (m1, m2, m3) in [(5, 5, 4), (5, 4, 5)]:
        t1 = (np.arange(m1) + 0.5) / m1 - 0.5
        t2 = (np.arange(m2) + 0.5) / m2 - 0.5
        t3 = (np.arange(m3) + 0.5) / m3 - 0.5
        g = np.array([(x, y, z) for z in t3 for y in t2 for x in t1])[:N]
        rng = np.random.RandomState(200 + m1)
        g = g + rng.uniform(-0.005, 0.005, size=g.shape)
        g = np.clip(g, -0.45, 0.45)
        attempts.append(g)
    
    for idx, initial in enumerate(attempts):
        if time.time() > deadline - 20:
            break
        
        relaxed, score = relax(initial, max_iters=400)
        
        if score > best_score:
            best_score = score
            best_centers = relaxed.copy()
    
    if best_centers is None:
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])[:N]
        return grid
    
    return best_centers
# EVOLVE-BLOCK-END
