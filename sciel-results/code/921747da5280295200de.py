import numpy as np
import time

N = 300


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    def get_score(points):
        """
        Computes the minimum distance between any two circles and the boundary.
        Returns -1 if invalid.
        """
        if len(points) != N:
            return -1.0
        if not np.all(np.isfinite(points)):
            return -1.0
        
        # Check boundary
        dists_center = np.linalg.norm(points, axis=1)
        if np.any(dists_center >= 1.0):
            return -1.0
        
        # Pairwise distances
        # Using broadcasting for speed, but careful with memory. N=300 is small.
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        
        min_pair_dist = np.min(d) / 2.0
        min_bound_dist = np.min(1.0 - dists_center)
        
        return min(min_pair_dist, min_bound_dist)

    def init_hexagonal():
        """
        Generates a hexagonal lattice that fits within the unit circle.
        Ensures we get at least N points.
        """
        # Estimated radius
        r_est = np.sqrt(0.9069 / N)
        spacing = 2 * r_est * 1.02 # Slightly larger spacing to ensure fit, we'll optimize
        
        # Generate grid
        # We need to cover the unit circle.
        # Max x, y is 1.
        # Number of points in one dimension approx 1/spacing + 1
        k = int(2.0 / spacing) + 5
        
        points_list = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                points_list.append([x, y])
        
        points = np.array(points_list)
        if points.size == 0:
            # Fallback to random if something went wrong with lattice generation
            return np.random.uniform(-1, 1, (N, 2))

        # Keep only points inside unit circle (with small margin)
        r = np.linalg.norm(points, axis=1)
        mask = r < 1.0
        points = points[mask]
        r = r[mask]
        
        if len(points) < N:
            # If we didn't get enough points, we need to expand the grid or use random
            # This is unlikely with k=5 expansion, but let's be safe.
            # Fallback: Random points inside circle
            angles = np.random.uniform(0, 2*np.pi, N)
            radii = np.sqrt(np.random.uniform(0, 1, N)) * 0.9
            points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        else:
            # Sort by distance from center to pick the central ones
            idx = np.argsort(r)
            points = points[idx[:N]]
            
        return points

    def optimize(points, max_iters=1000):
        """
        Gradient descent style optimization to maximize the minimum distance.
        Uses repulsive forces for close pairs and attractive force to center.
        """
        points = points.copy().astype(float)
        best_score = get_score(points)
        best_points = points.copy()
        
        # Parameters
        learning_rate = 0.01
        decay = 0.999
        force_exp = 2.0 # Force ~ 1/d^exp
        
        for it in range(max_iters):
            if time.time() > deadline - 5.0:
                break
            if it % 50 == 0 and it > 0:
                # Adaptive learning rate reduction based on progress or just time
                pass 
            
            # Calculate gradients (forces)
            # 1. Pairwise repulsion
            diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
            d = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(d, np.inf)
            
            # We want to maximize min(d). 
            # Approximation: apply force only to pairs closer than some threshold or all pairs with weight.
            # Force direction: away from neighbor.
            # Magnitude: proportional to 1/d^force_exp, but we only care about the minimum.
            # A common heuristic is to apply force to all pairs, weighted by inverse distance.
            
            # To avoid division by zero or huge forces, clip d
            d_safe = np.clip(d, 1e-6, np.inf)
            
            # Force magnitude
            # We want to push apart.
            # grad of min_d is tricky. Let's use a potential energy approach:
            # E = sum (1/d^2) for d < threshold, or just sum 1/d^2 for all.
            # Minimizing E maximizes distances roughly.
            
            weights = 1.0 / (d_safe ** force_exp)
            # Zero out diagonal
            np.fill_diagonal(weights, 0.0)
            
            # Force vector: sum of (diff / d) * weight
            # diff is (p_i - p_j). Force on i should be in direction (p_i - p_j) to push it away from j.
            # So F_i = sum_j (p_i - p_j) / d_ij * weight_ij
            # Note: diff is p_i - p_j.
            # Unit vector u_ij = diff / d
            # Force = u_ij * weight
            
            # We can compute this as: (diff / d_safe) * weights
            # But we need to handle the division carefully.
            # F_i = sum_j (p_i - p_j) * (1/d_safe^force_exp) / d_safe
            #      = sum_j (p_i - p_j) / d_safe^(force_exp + 1)
            
            # Let's compute the sum of forces
            # To save memory and time, we can do this in chunks or just rely on N=300 being small enough for full matrix.
            # 300x300x2 is ~ 180k floats, very fast.
            
            inv_d_pow = 1.0 / (d_safe ** (force_exp + 1))
            np.fill_diagonal(inv_d_pow, 0.0)
            
            forces = np.sum(diff * inv_d_pow[:, :, np.newaxis], axis=1)
            
            # 2. Boundary constraint
            # Force towards center if close to boundary
            dists_center = np.linalg.norm(points, axis=1)
            # Soft wall potential
            # If dist > 0.8, apply force towards center
            # F_wall = -k * (dist - 0.8) * (p / dist)
            # Or simpler: always pull towards center with small strength, stronger if outside.
            # Let's use a radial force: F = -points * alpha
            alpha = 0.01
            forces += -points * alpha
            
            # Normalize forces to prevent explosion
            force_norms = np.linalg.norm(forces, axis=1)
            max_norm = np.max(force_norms)
            if max_norm > 0:
                # Scale forces so max step is controlled
                forces = forces / max_norm * 0.1
            
            # Update
            points += forces * learning_rate
            
            # Project back to unit circle
            new_dists = np.linalg.norm(points, axis=1)
            mask = new_dists > 0.999
            if np.any(mask):
                # Reshape to (N, 1) to allow broadcasting with (N, 2)
                scale = (0.999 / new_dists[mask])[:, np.newaxis]
                points[mask] *= scale
            
            # Check score
            if it % 10 == 0:
                current_score = get_score(points)
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
            
            # Decay learning rate
            learning_rate *= decay
        
        return best_points, best_score

    # Initial setup
    np.random.seed(42) # For reproducibility in debugging, though evaluator might not care
    points = init_hexagonal()
    
    # Run main optimization until deadline is close
    # We pass a large max_iters, but the function checks time internally.
    best_points, best_score = optimize(points, max_iters=10000)
    
    # If we still have time, try a random restart
    if time.time() < deadline - 10.0:
        angles = np.random.uniform(0, 2*np.pi, N)
        radii = np.sqrt(np.random.uniform(0, 1, N)) * 0.9
        rand_points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        p_opt, s_opt = optimize(rand_points, max_iters=10000)
        if s_opt > best_score:
            best_score = s_opt
            best_points = p_opt
    
    # Final check
    final_score = get_score(best_points)
    if final_score < 0:
        # Fallback to a valid configuration
        # Simple grid or random that is definitely valid
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        radii = np.linspace(0, 0.9, N)
        best_points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        # Ensure no overlaps by shrinking if necessary? 
        # For N=300, a simple spiral might overlap.
        # Let's just return the best we have, trusting the optimizer fixed it.
        # If it's still invalid, we try a very conservative spiral.
        if get_score(best_points) < 0:
             # Conservative: place on a spiral with large spacing
             # This is a last resort
             theta = np.linspace(0, 20*np.pi, N)
             r = np.linspace(0, 0.95, N)
             best_points = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
             # Check if valid, if not, shrink
             while get_score(best_points) < 0 and time.time() < deadline - 1:
                 best_points *= 0.95

    return best_points
