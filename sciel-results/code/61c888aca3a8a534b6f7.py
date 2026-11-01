import numpy as np
import time

N = 50

def construct():
    """
    Constructs the centers of 50 equal circles in a unit square [-0.5, 0.5]^2
    to maximize the minimum distance between circles and to the boundary.
    Uses Best-In-First initialization followed by Simulated Annealing.
    """
    
    def compute_min_dist(points):
        """
        Computes the minimum distance between any two points and to the boundary.
        Returns (min_dist, min_pair_idx, min_wall_idx) where min_wall_idx is 
        (point_idx, wall_idx) or None.
        """
        # Pairwise distances
        # Using a loop to avoid memory overhead of cdist for small N, 
        # but cdist is fine too. Let's use a vectorized approach for speed if N is small.
        # Actually, for N=50, a simple loop or cdist is fast.
        # Let's use cdist for simplicity and speed in C.
        from scipy.spatial.distance import cdist
        dists = cdist(points, points)
        np.fill_diagonal(dists, np.inf)
        
        min_pair_val = np.inf
        min_pair_idx = (0, 0)
        
        # Find global min pair
        min_val, idx_flat = np.unravel_index(np.argmin(dists), dists.shape)
        min_pair_val = dists[min_val, idx_flat]
        min_pair_idx = (min_val, idx_flat)
        
        # Boundary distances
        # dist to left: 0.5 + x
        # dist to right: 0.5 - x
        # dist to down: 0.5 + y
        # dist to up: 0.5 - y
        
        # We can compute the min boundary distance for each point
        # d_i = min(0.5+xi, 0.5-xi, 0.5+yi, 0.5-yi)
        
        # Vectorized calculation
        d_left = 0.5 + points[:, 0]
        d_right = 0.5 - points[:, 0]
        d_down = 0.5 + points[:, 1]
        d_up = 0.5 - points[:, 1]
        
        # Stack them to find min per point and which wall
        # Shape (N, 4)
        walls = np.stack([d_left, d_right, d_down, d_up], axis=1)
        
        min_wall_val_per_point = np.min(walls, axis=1)
        min_wall_idx_per_point = np.argmin(walls, axis=1)
        
        # Global min boundary
        global_min_wall_point_idx = np.argmin(min_wall_val_per_point)
        min_wall_val = min_wall_val_per_point[global_min_wall_point_idx]
        min_wall_wall_idx = min_wall_idx_per_point[global_min_wall_point_idx]
        
        min_wall_idx = (global_min_wall_point_idx, min_wall_wall_idx)
        
        # Compare pair and wall
        if min_pair_val < min_wall_val:
            return min_pair_val, min_pair_idx, None
        else:
            return min_wall_val, None, min_wall_idx

    def best_in_first_init():
        """
        Initializes points using the Best-In-First heuristic.
        """
        points = np.zeros((N, 2))
        
        # Place first point at center
        points[0] = [0.0, 0.0]
        
        for i in range(1, N):
            # Generate candidate points on a grid
            # Step size roughly 1/20 or so
            step = 1.0 / 40.0
            candidates = []
            for x in np.arange(-0.5, 0.5 + step, step):
                for y in np.arange(-0.5, 0.5 + step, step):
                    candidates.append([x, y])
            candidates = np.array(candidates)
            
            best_score = -1
            best_point = None
            
            for p in candidates:
                # Check distance to existing points
                dists_to_existing = np.linalg.norm(points[:i] - p, axis=1)
                min_dist_existing = np.min(dists_to_existing) if i > 0 else np.inf
                
                # Check distance to boundary
                dist_to_left = 0.5 + p[0]
                dist_to_right = 0.5 - p[0]
                dist_to_down = 0.5 + p[1]
                dist_to_up = 0.5 - p[1]
                min_dist_boundary = min(dist_to_left, dist_to_right, dist_to_down, dist_to_up)
                
                score = min(min_dist_existing, min_dist_boundary)
                
                if score > best_score:
                    best_score = score
                    best_point = p
            
            points[i] = best_point
        
        return points

    # 1. Initialization
    points = best_in_first_init()
    
    # 2. Simulated Annealing
    # Parameters
    T = 0.01  # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.999
    max_iter = 10000
    current_score, _, _ = compute_min_dist(points)
    best_score = current_score
    best_points = points.copy()
    
    # Pre-compute random directions? No, generate on fly.
    
    for i in range(max_iter):
        # Generate a random move
        # Pick a random point
        idx = np.random.randint(N)
        
        # Pick a random direction and magnitude
        # Magnitude scaled by temperature
        angle = np.random.uniform(0, 2 * np.pi)
        magnitude = np.random.uniform(0, T * 10) # Scale factor
        
        dx = magnitude * np.cos(angle)
        dy = magnitude * np.sin(angle)
        
        new_points = points.copy()
        new_points[idx, 0] += dx
        new_points[idx, 1] += dy
        
        # Check if point is inside square
        if new_points[idx, 0] < -0.5 or new_points[idx, 0] > 0.5 or new_points[idx, 1] < -0.5 or new_points[idx, 1] > 0.5:
            continue
            
        # Compute new score
        new_score, _, _ = compute_min_dist(new_points)
        
        # Acceptance criterion
        if new_score > current_score:
            points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
        else:
            # Boltzmann acceptance
            # P = exp((new_score - current_score) / T)
            # Note: scores are distances, so we want to maximize.
            # Delta = new - current < 0
            delta = new_score - current_score
            if delta < 0:
                # Avoid overflow in exp
                if delta / T > -500:
                    p = np.exp(delta / T)
                else:
                    p = 0
                
                if np.random.random() < p:
                    points = new_points
                    current_score = new_score
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            break
            
    return best_points

# EVOLVE-BLOCK-END
