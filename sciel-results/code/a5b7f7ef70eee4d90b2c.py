import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    """
    Uses Simulated Annealing to find a near-optimal packing of N circles in a unit circle.
    Initializes with a multi-ring heuristic which is structurally closer to optimal packings.
    """
    
    def calculate_radius(points):
        """
        Calculate the maximum radius for the given points.
        r = min(min_pairwise_dist / 2, min_boundary_dist)
        """
        # Pairwise distances
        # Using broadcasting for efficiency
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pairwise = np.min(dists)
        
        # Boundary distances
        radii = np.linalg.norm(points, axis=1)
        min_boundary = 1.0 - np.max(radii)
        
        return min(min_pairwise / 2.0, min_boundary)

    def initialize_multi_ring():
        """
        Initialize points in concentric rings.
        For N=50, a good configuration is 1 center + rings of 6, 12, 18, 13? 
        Let's try to fill rings as evenly as possible.
        Ring 0: 1
        Ring 1: 6
        Ring 2: 12
        Ring 3: 18
        Ring 4: 13 (remaining)
        Total: 1+6+12+18+13 = 50.
        
        We estimate radii to be roughly proportional to sqrt(i) to maintain density,
        but for a starting point, we can just spread them out.
        """
        points = np.zeros((N, 2))
        
        # Define ring sizes
        ring_sizes = [1, 6, 12, 18, 13]
        
        idx = 0
        for ring_idx, size in enumerate(ring_sizes):
            if ring_idx == 0:
                # Center
                points[idx] = [0.0, 0.0]
                idx += 1
            else:
                # Approximate radius for this ring
                # We want to fit 'size' circles in a ring.
                # Circumference ~ size * (2 * r_est)
                # Area ~ pi * (R_outer^2 - R_inner^2)
                # Let's just guess radii based on index.
                # A simple heuristic: r_i ~ sqrt(i * N / pi)
                # But we need to place them specifically.
                # Let's use a radius that increases with ring index.
                # For N=50, max radius is 1.
                # Ring 1 (6 circles): r ~ 0.3
                # Ring 2 (12 circles): r ~ 0.5
                # Ring 3 (18 circles): r ~ 0.7
                # Ring 4 (13 circles): r ~ 0.9
                # These are rough estimates. The SA will adjust them.
                
                # Better heuristic: 
                # The area of the disk is 1. 
                # If we distribute N points uniformly, the k-th point (sorted by radius) 
                # has radius approx sqrt(k/N).
                # Here we are grouping by rings.
                
                # Let's just pick increasing radii.
                # r_1 = 0.3, r_2 = 0.55, r_3 = 0.75, r_4 = 0.9
                radii_guess = [0.0, 0.3, 0.55, 0.75, 0.9]
                r = radii_guess[ring_idx]
                
                for k in range(size):
                    theta = 2 * np.pi * k / size
                    # Add a small random jitter to break symmetry and help SA
                    theta += np.random.uniform(-0.05, 0.05)
                    points[idx] = [r * np.cos(theta), r * np.sin(theta)]
                    idx += 1
        
        return points

    # Start with multi-ring initialization
    current_points = initialize_multi_ring()
    current_radius = calculate_radius(current_points)
    
    best_points = current_points.copy()
    best_radius = current_radius
    
    # Simulated Annealing parameters
    T = 0.05  # Initial temperature
    T_min = 1e-8
    cooling_rate = 0.9995
    max_iters = 200000 # High number, we will stop by time
    
    start_time = time.time()
    time_limit = 115.0
    
    # Pre-compute some constants or use vectorized ops where possible
    # But for SA, we need to evaluate a single move quickly.
    # We can maintain the pairwise distance matrix or just recalculate?
    # Recalculating N=50 pairwise dists is 1225 ops. 
    # 200k iters * 1225 ops = 245M ops. In Python/NumPy this might be slow if done naively in a loop.
    # However, we can optimize the evaluation of a single move.
    
    # Let's try to keep it simple first. If it's too slow, we optimize.
    # NumPy vectorized distance calculation is quite fast.
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        # Pick a random point to move
        i = np.random.randint(N)
        
        # Save old position
        old_pos = current_points[i].copy()
        
        # Generate new position
        # Random walk with Gaussian steps
        # Step size should be related to current temperature
        step_size = T * 0.1
        new_pos = old_pos + np.random.normal(0, step_size, 2)
        
        # Ensure the new point is not completely outside the container?
        # It's okay if it is, the radius calculation will handle it (boundary term becomes negative or small).
        # But to be efficient, we can reject if it's way out.
        if np.linalg.norm(new_pos) > 1.5:
            continue
            
        current_points[i] = new_pos
        
        # Calculate new radius
        new_radius = calculate_radius(current_points)
        
        # Calculate delta
        delta = new_radius - current_radius
        
        # Acceptance probability
        if delta > 0:
            # Always accept if better
            current_radius = new_radius
            if new_radius > best_radius:
                best_radius = new_radius
                best_points = current_points.copy()
        else:
            # Accept with probability exp(delta / T)
            # delta is negative, so exp(delta/T) < 1
            # To avoid overflow/underflow issues with very negative delta/T
            # If delta/T is very negative, prob is 0.
            if T > 0:
                prob = np.exp(delta / T)
            else:
                prob = 0.0
                
            if np.random.rand() < prob:
                current_radius = new_radius
            else:
                # Reject move, revert
                current_points[i] = old_pos
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            # If we hit T_min, we might restart with a slightly higher T to escape local minima?
            # Or just continue.
            # Let's add a simple restart mechanism if stuck for a long time.
            # For now, just continue.
            
    # Final refinement: 
    # The SA might end in a slightly suboptimal local minimum.
    # We can run a few iterations of "hill climbing" (only accepting better moves) 
    # or just return the best found.
    # Given the time limit, returning the best found is safe.
    
    # One final check to ensure validity
    # The score is calculated by the evaluator, but we should ensure no NaNs or infinite values.
    if not np.all(np.isfinite(best_points)):
        # Fallback to a simple grid if something went wrong
        best_points = np.zeros((N, 2))
        for i in range(N):
            r_init = np.sqrt(i / N) * 0.9
            theta = i * np.pi * (3 - np.sqrt(5))
            best_points[i] = [r_init * np.cos(theta), r_init * np.sin(theta)]
            
    return best_points

# EVOLVE-BLOCK-END
