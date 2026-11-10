"""Improved packing of 300 equal circles in a unit circle using local optimization."""
import numpy as np
import time
import math

N = 300

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    def initial_hex(n):
        """Generate initial hexagonal lattice points."""
        # Estimate spacing
        # Area per point ~ sqrt(3)/2 * s^2
        # N * sqrt(3)/2 * s^2 ~ pi * (1-r)^2
        # s ~ sqrt(2 * pi * (1-r)^2 / (N * sqrt(3)))
        # r ~ 0.05, 1-r ~ 0.95
        s_est = math.sqrt(2 * math.pi * 0.95**2 / (n * math.sqrt(3)))
        
        best_points = None
        best_score = -1
        
        # Try a few spacings around the estimate
        for trial in range(20):
            spacing = s_est * (1.0 + trial * 0.005)
            points = []
            r_max = int(1.0 / (spacing * math.sqrt(3) / 2)) + 2
            for j in range(-r_max, r_max + 1):
                for i in range(-r_max, r_max + 1):
                    x = (i + 0.5 * (j % 2)) * spacing
                    y = j * spacing * math.sqrt(3) / 2
                    if x * x + y * y <= (1.0 - spacing / 2) ** 2:
                        points.append((x, y))
            
            if len(points) >= n:
                points_arr = np.array(points)
                dists = np.hypot(points_arr[:, 0], points_arr[:, 1])
                idx = np.argsort(dists)[:n]
                candidate = points_arr[idx]
                
                # Quick score check
                score = compute_score(candidate)
                if score > best_score:
                    best_score = score
                    best_points = candidate
            
            if time.time() > deadline - 5:
                break
        
        if best_points is None:
            # Fallback: random
            best_points = np.random.randn(n, 2) * 0.8
            # Scale to fit
            max_r = np.max(np.hypot(best_points[:, 0], best_points[:, 1]))
            if max_r > 0.9:
                best_points *= 0.9 / max_r
        return best_points

    def compute_score(points):
        """Compute the minimum radius for the entire configuration (O(N^2))."""
        # Vectorized pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        radii = np.linalg.norm(points, axis=1)
        min_boundary = np.min(1.0 - radii)
        
        return min(min_dist / 2.0, min_boundary)

    def get_point_min_radius(points, i):
        """Compute the minimum radius contribution for point i vs others and boundary (O(N))."""
        # Distance to all other points
        diffs = points - points[i]
        dists = np.linalg.norm(diffs, axis=1)
        dists[i] = np.inf # Ignore self
        min_dist_to_others = np.min(dists)
        
        # Distance to boundary
        r_i = np.linalg.norm(points[i])
        min_boundary = 1.0 - r_i
        
        return min(min_dist_to_others / 2.0, min_boundary)

    def local_optimize(points, max_iter=500):
        """Fast local optimization using O(N) score for single point updates."""
        points = points.copy()
        current_score = compute_score(points)
        best_points = points.copy()
        best_score = current_score
        
        lr = 0.01
        no_improve_count = 0
        
        for it in range(max_iter):
            if time.time() > deadline - 2.0: # Safety margin
                break
            
            improved = False
            # Shuffle indices for stochasticity
            indices = np.random.permutation(N)
            
            for i in indices:
                if time.time() > deadline - 2.0:
                    break
                
                # Try moving point i in a random direction
                # Use 2 attempts per point to balance speed and convergence
                for _ in range(2):
                    angle = np.random.uniform(0, 2 * np.pi)
                    dx = lr * math.cos(angle)
                    dy = lr * math.sin(angle)
                    
                    old_x, old_y = points[i]
                    new_x, new_y = old_x + dx, old_y + dy
                    
                    # Check bounds
                    r_new = math.hypot(new_x, new_y)
                    if r_new > 0.999:
                        scale = 0.999 / r_new
                        new_x *= scale
                        new_y *= scale
                    
                    # Calculate new min radius for this point only (O(N))
                    # We approximate the global score change by looking at the limiting factor for point i
                    # However, moving i might affect the min distance for other points j if i was the closest neighbor to j.
                    # To be safe and fast, we check if the new point is valid and improves the local constraint.
                    # A more accurate but slightly slower check:
                    # 1. Save old point
                    # 2. Update point
                    # 3. Compute min radius for point i (O(N))
                    # 4. If this new local min is > current global min, it might improve.
                    # But we don't know if it breaks a constraint elsewhere.
                    # Simplest robust fast check:
                    # Update point in array, compute global score? No, too slow.
                    
                    # Heuristic: Only accept if the new position is not worse than the current limiting factor for i,
                    # and we periodically recompute global score.
                    
                    # Let's do a strict check:
                    # Temporarily update
                    points[i] = [new_x, new_y]
                    
                    # Check if this move violates the current global minimum radius?
                    # If current_score is determined by point i or a neighbor, this is tricky.
                    # Let's just compute the min radius for point i.
                    local_min_r = get_point_min_radius(points, i)
                    
                    # If local_min_r is significantly larger than current_score, it's a good candidate.
                    # If local_min_r < current_score, it might have broken something, but usually 
                    # random moves in dense packing are constrained by local neighbors.
                    
                    # To be safe, we accept if local_min_r >= current_score * 0.99 (allowing slight noise)
                    # OR if it improves the local bottleneck.
                    # Actually, the most reliable fast method is to check the new distance to the closest neighbor.
                    
                    # Let's revert and do a simpler check:
                    points[i] = [old_x, old_y]
                    
                    # Compute distance to closest neighbor for new pos
                    diffs = points - np.array([new_x, new_y])
                    dists = np.linalg.norm(diffs, axis=1)
                    dists[i] = np.inf
                    closest_dist = np.min(dists)
                    
                    # Boundary distance
                    r_new = math.hypot(new_x, new_y)
                    bound_dist = 1.0 - r_new
                    
                    new_local_r = min(closest_dist / 2.0, bound_dist)
                    
                    # Accept if it improves the current global score?
                    # We don't know the global score without O(N^2).
                    # Strategy: Accept if new_local_r > current_score.
                    # This ensures we don't make it worse.
                    if new_local_r > current_score:
                        points[i] = [new_x, new_y]
                        # Update current_score? 
                        # The global score might have increased if the limiting factor was point i.
                        # If the limiting factor was elsewhere, current_score stays same.
                        # We can't update current_score accurately without O(N^2).
                        # So we just keep moving. We will recompute global score every 100 iterations or when stuck.
                        improved = True
                        break # Move to next point
            
            # Recompute global score periodically to update best_score and current_score
            if it % 50 == 0 or not improved:
                current_score = compute_score(points)
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    lr *= 0.95
                    if no_improve_count > 10 or lr < 1e-5:
                        break
            else:
                no_improve_count = 0
        
        # Final full score computation
        final_score = compute_score(points)
        if final_score > best_score:
            best_score = final_score
            best_points = points.copy()
            
        return best_points, best_score

    # Generate initial configuration
    points = initial_hex(N)
    initial_score = compute_score(points)
    
    # Local optimization
    points, score = local_optimize(points, max_iter=200)
    
    # If time remains, try a second round with random init
    if time.time() < deadline - 30:
        np.random.seed(42)
        points2 = np.random.randn(N, 2) * 0.7
        max_r = np.max(np.hypot(points2[:, 0], points2[:, 1]))
        if max_r > 0.95:
            points2 *= 0.95 / max_r
        
        points2, score2 = local_optimize(points2, max_iter=200)
        if score2 > score:
            points = points2
            score = score2
    
    # Final check
    if time.time() < deadline - 5:
        # One more polish
        points, score_final = local_optimize(points, max_iter=50)
        if score_final > score:
            points = points
            score = score_final
            
    return points
# EVOLVE-BLOCK-END
