import numpy as np
import time

N = 200

def construct():
    """
    Packs N equal circles in a unit square [-0.5, 0.5]^2 to maximize radius.
    Uses a Best-First Local Search with Shaking.
    """
    start_time = time.time()
    time_limit = 110.0  # Safety margin for 120s limit
    
    # 1. Initialization: Grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    # Create full grid and take first N
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centers = grid[:N].copy()
    
    # Add small jitter to break symmetry
    np.random.seed(42)
    centers += np.random.normal(0, 0.001, centers.shape)
    
    # Ensure inside bounds
    centers = np.clip(centers, -0.5, 0.5)
    
    def calc_radius(centers):
        """Calculate the maximum possible radius for the given centers."""
        # Distance to boundary
        dists = np.min(np.minimum(0.5 - np.abs(centers), 0.5 - np.abs(centers)), axis=1)
        min_boundary = np.min(dists)
        
        # Distance between circles
        # For N=200, O(N^2) is 40,000 ops, which is fast.
        # We can vectorize this.
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        min_pair_dist = np.min(np.sqrt(dists_sq))
        
        return min(min_boundary, min_pair_dist / 2.0)
    
    current_radius = calc_radius(centers)
    best_centers = centers.copy()
    best_radius = current_radius
    
    stuck_count = 0
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Best-First Search: Try moving each circle to its best local position
        best_gain = 0
        best_move_idx = -1
        best_new_pos = None
        
        # To speed up, we might not check all 200 every time if it's slow, 
        # but N=200 is small. Let's check all.
        for i in range(N):
            # Find nearest neighbor
            dists = np.sum((centers - centers[i])**2, axis=1)
            dists[i] = np.inf
            j = np.argmin(dists)
            
            # Direction to move away from j
            direction = centers[i] - centers[j]
            norm = np.linalg.norm(direction)
            if norm < 1e-9:
                continue
            direction /= norm
            
            # 1D search along this direction
            # We want to maximize min(dist to other circles, dist to boundary)
            # Let's try a few steps
            best_local_radius = -np.inf
            best_step = 0
            
            # Search range: up to the distance to the nearest neighbor
            max_step = np.sqrt(dists[j]) * 0.99
            steps = np.linspace(0, max_step, 20)
            
            for step in steps:
                new_pos = centers[i] + step * direction
                # Check if new_pos is inside boundary
                if np.any(np.abs(new_pos) > 0.5):
                    continue
                
                # Calculate new radius if center i moves to new_pos
                # Dist to boundary
                new_dist_boundary = np.min(0.5 - np.abs(new_pos))
                
                # Dist to other circles
                # Only need to check distance to other circles, not all pairs
                # But the global radius is min(all pairs).
                # If we move i, the only new distances are i to k (k!=i).
                # The other distances remain the same.
                # So new global radius = min(old_min_pair_dist_except_i, new_min_dist_i)
                
                # This is tricky. Let's just calculate the full radius for the candidate.
                # It's O(N) for distance to others, plus O(1) for boundary.
                # Total O(N) per candidate. 20 candidates * 200 circles = 4000 ops per i.
                # 200 * 4000 = 800,000 ops per iteration.
                # 1000 iterations in 110s -> ~8M ops/s. Should be fine in NumPy.
                
                temp_centers = centers.copy()
                temp_centers[i] = new_pos
                r = calc_radius(temp_centers)
                
                if r > best_local_radius:
                    best_local_radius = r
                    best_step = step
            
            if best_local_radius > current_radius + best_gain:
                best_gain = best_local_radius - current_radius
                best_move_idx = i
                best_new_pos = centers[best_move_idx] + best_step * direction
        
        if best_move_idx != -1 and best_gain > 1e-12:
            centers[best_move_idx] = best_new_pos
            current_radius += best_gain
            improved = True
            stuck_count = 0
            
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = centers.copy()
        else:
            stuck_count += 1
            
            # If stuck, shake
            if stuck_count > 100:
                # Randomly perturb a few circles
                num_shake = 5
                idxs = np.random.choice(N, num_shake, replace=False)
                for idx in idxs:
                    centers[idx] += np.random.normal(0, 0.01, 2)
                    centers[idx] = np.clip(centers[idx], -0.5, 0.5)
                
                current_radius = calc_radius(centers)
                stuck_count = 0
                
                if current_radius > best_radius:
                    best_radius = current_radius
                    best_centers = centers.copy()
    
    return best_centers
