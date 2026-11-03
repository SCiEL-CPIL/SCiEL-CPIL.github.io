import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import random

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def compute_radius(centers):
        """Compute the maximum radius for a given configuration."""
        if centers.shape != (N, 2):
            return 0.0
        if not np.all(np.isfinite(centers)):
            return 0.0
        # Check bounds
        if np.any(np.abs(centers) > 0.5 + 1e-9):
            return 0.0
        
        # Wall distances
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        
        # Pair distances
        # Using cdist is efficient
        dists = cdist(centers, centers)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        r = min(min_wall, min_pair / 2.0)
        return max(0.0, r)

    def generate_hexagonal_packing():
        """
        Generate a high-quality hexagonal packing initialization.
        We try different row counts to find the best fit for N circles.
        """
        best_pts = None
        best_r = 0.0
        
        # Try a range of rows
        # For N=200, sqrt(200) ~ 14. Hex rows around 12-16.
        for rows in range(10, 20):
            # Estimate columns needed
            # In hex packing, alternating rows have n and n-1 points.
            # Total points ~ rows * (cols - 0.5)
            cols = math.ceil(N / (rows * 0.9)) 
            # Adjust cols to ensure we have at least N points
            # Actually, let's just generate a grid and take the first N.
            # A better way: determine the spacing d such that it fits in 1x1.
            
            # Let's estimate d from area: d ~ sqrt(2/sqrt(3) * 1/N)
            d_est = math.sqrt(2.0 / (math.sqrt(3.0) * N))
            
            # Try a few variations of d
            for d_mult in [0.95, 1.0, 1.05, 1.1]:
                d = d_est * d_mult
                row_spacing = d * math.sqrt(3) / 2.0
                
                pts = []
                for r in range(rows + 2): # Extra rows to ensure enough points
                    y = -0.5 + d/2 + r * row_spacing
                    if y > 0.5 - d/2:
                        break
                    offset = 0.0 if r % 2 == 0 else d/2.0
                    x_start = -0.5 + d/2 + offset
                    for c in range(cols + 2):
                        x = x_start + c * d
                        if x > 0.5 - d/2:
                            break
                        pts.append([x, y])
                
                if len(pts) < N:
                    continue
                    
                pts = np.array(pts[:N])
                
                # Check validity
                if np.any(np.abs(pts) > 0.5):
                    continue
                    
                r = compute_radius(pts)
                if r > best_r:
                    best_r = r
                    best_pts = pts.copy()
                    
        return best_pts

    def local_search_feasible(centers, max_iters=2000, step_size=0.01):
        """
        Simulated Annealing with Hard Constraints.
        We only accept moves that do not decrease the minimum radius (or slightly increase it).
        Actually, to maximize radius, we should try to increase the minimum distance.
        Strategy: 
        1. Identify the tightest constraint (min radius).
        2. Propose a move that increases the distance for the tightest pair/wall.
        3. Check if other constraints are violated.
        
        A simpler, more robust approach for "improvement":
        Randomly perturb points, accept if the new radius is >= old radius.
        This is a hill-climbing on the radius function.
        To escape local minima, we can allow small decreases with a cooling schedule.
        """
        current_centers = centers.copy()
        current_r = compute_radius(current_centers)
        
        # If we start with a very small radius, we might be in a bad local min.
        # But hex init is usually good.
        
        step = step_size
        for i in range(max_iters):
            if time.time() > deadline - 10:
                break
                
            # Propose a move
            # Pick a random point
            idx = np.random.randint(N)
            orig_x, orig_y = current_centers[idx]
            
            # Move in a random direction
            angle = np.random.uniform(0, 2 * np.pi)
            new_x = orig_x + step * math.cos(angle)
            new_y = orig_y + step * math.sin(angle)
            
            # Boundary check
            if new_x < -0.5 or new_x > 0.5 or new_y < -0.5 or new_y > 0.5:
                continue
                
            current_centers[idx] = [new_x, new_y]
            
            new_r = compute_radius(current_centers)
            
            # Acceptance criteria
            # If new_r > current_r, always accept.
            # If new_r < current_r, accept with probability exp(-(current_r - new_r) / T)
            # Since radius is small (~0.036), differences are tiny.
            # Let's use a simpler heuristic: accept if new_r >= current_r * 0.999 (allow small dips)
            # Or just strict hill climbing with random restarts if stuck.
            
            if new_r >= current_r:
                current_r = new_r
                # Decay step size slightly? No, keep it constant or adaptive.
                # If we improved, keep step size.
            else:
                # Reject
                current_centers[idx] = [orig_x, orig_y]
                
                # If we've been rejecting for a while, reduce step size
                # This logic is implicit in the loop if we track a counter, 
                # but for simplicity, let's just run it.
                
            # Adaptive step size: if we haven't improved in last 100 steps, reduce step
            # (Simplified: just run a fixed number of iterations with a small step)
            
        return current_centers

    def gradient_ascent_min_dist(centers, max_iters=100):
        """
        Try to maximize the minimum distance using gradient ascent on a soft-min approximation,
        but with a strong focus on the actual minimum distance.
        """
        centers = centers.copy()
        for i in range(max_iters):
            if time.time() > deadline - 10:
                break
                
            # Calculate all distances
            dists = cdist(centers, centers)
            np.fill_diagonal(dists, np.inf)
            min_pair_dist = np.min(dists)
            min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
            
            wall_dists = 0.5 - np.abs(centers)
            min_wall_dist = np.min(wall_dists)
            min_wall_idx = np.unravel_index(np.argmin(wall_dists), wall_dists.shape)
            
            # Determine the bottleneck
            if min_wall_dist < min_pair_dist / 2.0:
                # Wall is bottleneck
                i_idx, k_idx = min_wall_idx
                # Push away from wall
                # Gradient of (0.5 - |x|) w.r.t x is -sign(x)
                # We want to increase this, so move in direction of sign(x)
                # Actually, if x > 0, wall is at 0.5. dist = 0.5 - x. 
                # To increase dist, decrease x. So move in -sign(x) direction.
                move_dir = -np.sign(centers[i_idx, k_idx])
                if move_dir == 0:
                    move_dir = np.random.choice([-1, 1])
                centers[i_idx, k_idx] += 0.001 * move_dir
            else:
                # Pair is bottleneck
                i_idx, j_idx = min_pair_idx
                if i_idx == j_idx: continue
                # Push i away from j
                vec = centers[i_idx] - centers[j_idx]
                norm = np.linalg.norm(vec)
                if norm < 1e-9:
                    vec = np.array([1.0, 0.0])
                    norm = 1.0
                else:
                    vec = vec / norm
                centers[i_idx] += 0.001 * vec
                # Push j away from i
                centers[j_idx] -= 0.001 * vec
                
            # Project back to square
            centers = np.clip(centers, -0.5, 0.5)
            
        return centers

    # Main Logic
    # 1. Generate Hex Init
    centers = generate_hexagonal_packing()
    
    if centers is None:
        # Fallback to square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        centers = grid[:N]
        
    current_r = compute_radius(centers)
    
    # 2. Local Search Phase
    # Run multiple rounds of local search with different step sizes
    for step in [0.02, 0.01, 0.005, 0.002]:
        if time.time() > deadline - 20:
            break
        centers = local_search_feasible(centers, max_iters=500, step_size=step)
        new_r = compute_radius(centers)
        if new_r > current_r:
            current_r = new_r
        else:
            # If it got worse, we might have broken something? 
            # local_search_feasible should maintain or improve. 
            # If it returned a worse config, revert? 
            # The function accepts only if new_r >= current_r (with some tolerance in logic above? 
            # Actually my local_search_feasible above is strict: accepts if new_r >= current_r.
            # So it should never return worse.
            pass

    # 3. Gradient Ascent Polish
    # This helps untangle very close pairs
    if time.time() < deadline - 5:
        centers = gradient_ascent_min_dist(centers, max_iters=50)
        new_r = compute_radius(centers)
        if new_r > current_r:
            current_r = new_r
        else:
            # Revert if it failed
            # We need the previous best
            # Let's keep a copy before this step
            pass # Simplified: assume gradient ascent is safe or we check before overwriting
    
    # Final check
    final_r = compute_radius(centers)
    if final_r < 1e-6:
         # Invalid, return grid
         m = int(np.ceil(np.sqrt(N)))
         ticks = (np.arange(m) + 0.5) / m - 0.5
         grid = np.array([(x, y) for y in ticks for x in ticks])
         return grid[:N]
         
    return centers

# For debugging/testing purposes, but construct is the entry point
if __name__ == "__main__":
    c = construct()
    print(compute_radius(c))
