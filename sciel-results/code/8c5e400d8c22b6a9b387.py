"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import cKDTree

N = 50

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    time_limit = 110.0 # Safety margin before 120s kill

    # 1. Initialization: Hexagonal Grid
    # This provides a valid, reasonably packed starting point.
    def get_hex_points(spacing):
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        # Filter points that fit in circle of radius 1 - spacing/2
        # This is a rough estimate for the radius.
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        radius = radius[mask]
        if len(points) >= N:
            # Sort by distance to center to get the most central ones
            idx = np.argsort(radius)
            return points[idx[:N]]
        return None

    # Find a good initial spacing
    spacing = 0.25
    best_init = None
    while spacing > 0.05:
        pts = get_hex_points(spacing)
        if pts is not None:
            best_init = pts
            break
        spacing *= 0.95
        
    if best_init is None:
        # Fallback: simple grid or random
        best_init = np.random.uniform(-0.5, 0.5, (N, 2))
        # Ensure they are inside
        norms = np.linalg.norm(best_init, axis=1)
        best_init[norms > 0.8] = best_init[norms > 0.8] * (0.8 / norms[norms > 0.8])[:, np.newaxis]

    current_centers = best_init.copy()

    # Helper: Calculate radius efficiently
    # For N=50, O(N^2) vectorized is fine, but cKDTree is O(N log N) and faster for repeated checks if we can use it.
    # However, for a single evaluation, vectorized numpy is often faster than building a tree.
    # Let's use vectorized for simplicity and speed at N=50.
    
    def calc_min_radius(centers):
        # Vectorized pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Boundary constraint
        norms = np.linalg.norm(centers, axis=1)
        min_boundary = 1.0 - np.max(norms)
        
        return min(min_pair_dist / 2.0, min_boundary)

    current_r = calc_min_radius(current_centers)
    best_centers = current_centers.copy()
    best_r = current_r

    # 2. Optimization: Simulated Annealing
    # We will run SA for the majority of the time budget.
    # Key: Efficient evaluation and good move generation.
    
    T = 0.01 # Initial temperature
    alpha = 0.9999 # Cooling factor
    T_min = 1e-5
    
    # Pre-allocate for speed? Not really needed for N=50.
    
    # Estimate iterations per second
    t0 = time.time()
    iters_test = 100
    for _ in range(iters_test):
        i = np.random.randint(N)
        delta = np.random.normal(0, T, 2)
        new_centers = current_centers.copy()
        new_centers[i] += delta
        r = calc_min_radius(new_centers)
    t_elapsed = time.time() - t0
    iters_per_sec = iters_test / t_elapsed if t_elapsed > 0 else 1000.0
    
    # Reserve time for final polish (e.g., 10%)
    polish_time_budget = 5.0
    sa_time_budget = time_limit - 5.0 - polish_time_budget
    
    # Calculate how many iterations we can afford
    # But we must also allow for the actual loop overhead.
    # Let's just run until time is up.
    
    t_start = time.time()
    max_sa_time = sa_time_budget
    
    # To improve convergence, we can use a "restart" strategy or adaptive step size.
    # Simple SA with fixed cooling is okay, but adaptive is better.
    # Let's stick to simple SA but with a slightly higher T to explore more initially.
    
    T = 0.02
    alpha = 0.9998
    
    import time as _time
    for it in range(10000000): # Large cap, time will stop it
        if time.time() - t_start > max_sa_time or time.time() - start_time > time_limit - 2.0:
            break
            
        # Pick a random center
        i = np.random.randint(N)
        
        # Move size proportional to T
        delta = np.random.normal(0, T, 2)
        
        # Optimization: Only update the specific center
        old_center = current_centers[i].copy()
        new_centers = current_centers.copy()
        new_centers[i] = old_center + delta
        
        new_r = calc_min_radius(new_centers)
        
        if new_r > current_r:
            current_centers = new_centers
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_centers = current_centers.copy()
        else:
            delta_r = current_r - new_r
            # Accept if delta_r is small or random prob
            # Use T for acceptance
            if np.random.rand() < np.exp(-delta_r / T):
                current_centers = new_centers
                current_r = new_r
                
        T *= alpha
        if T < T_min:
            T = T_min

    # 3. Final Polish: Local Optimization
    # Use a more deterministic local search to squeeze out the last bits.
    # We can try moving each point slightly in the direction that increases the minimum distance to its neighbors.
    # Or simply run a few more SA iterations with very low T.
    
    T = 1e-4
    alpha = 0.999
    t_start_polish = time.time()
    
    for it in range(100000):
        if time.time() - t_start_polish > polish_time_budget:
            break
        # Also check global limit
        if time.time() - start_time > time_limit - 2.0:
            break
            
        i = np.random.randint(N)
        # Smaller moves
        delta = np.random.normal(0, T, 2)
        
        new_centers = current_centers.copy()
        new_centers[i] = current_centers[i] + delta
        
        new_r = calc_min_radius(new_centers)
        
        if new_r > current_r:
            current_centers = new_centers
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_centers = current_centers.copy()
        else:
            delta_r = current_r - new_r
            if np.random.rand() < np.exp(-delta_r / T):
                current_centers = new_centers
                current_r = new_r
                
        T *= alpha
        if T < 1e-6:
            T = 1e-6

    # Final check
    final_r = calc_min_radius(best_centers)
    if final_r > best_r:
        best_r = final_r

    return best_centers

# EVOLVE-BLOCK-END
