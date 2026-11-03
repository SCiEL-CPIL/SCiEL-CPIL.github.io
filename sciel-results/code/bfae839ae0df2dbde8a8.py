import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    """
    Uses Simulated Annealing with repulsive moves to find a near-optimal packing of N circles in a unit circle.
    """
    
    def get_score(points):
        """
        Calculate the actual score: max common radius.
        """
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pairwise = np.min(dists)
        
        radii = np.linalg.norm(points, axis=1)
        min_boundary = 1.0 - np.max(radii)
        
        return min(min_pairwise / 2.0, min_boundary)

    # Initialization: Fermat spiral is a good dense initial packing
    points = np.zeros((N, 2))
    for i in range(N):
        r = np.sqrt(i / N) * 0.95  # Scale to fit well inside
        theta = i * np.pi * (3 - np.sqrt(5))
        points[i] = [r * np.cos(theta), r * np.sin(theta)]
    
    current_score = get_score(points)
    best_score = current_score
    best_points = points.copy()
    
    # SA Parameters
    T = 0.01
    T_min = 1e-8
    alpha = 0.9995
    step_size = 0.05
    
    start_time = time.time()
    time_limit = 115.0 # Leave 5s buffer
    
    while time.time() - start_time < time_limit:
        # Generate a new candidate using a repulsive move
        idx = np.random.randint(N)
        
        # Find the closest constraint for point idx
        # 1. Distance to boundary
        r_idx = np.linalg.norm(points[idx])
        dist_boundary = 1.0 - r_idx
        
        # 2. Distance to closest neighbor
        diffs = points - points[idx]
        dists = np.linalg.norm(diffs, axis=1)
        dists[idx] = np.inf
        closest_idx = np.argmin(dists)
        dist_closest = dists[closest_idx]
        
        # Determine direction to move
        # We want to increase the minimum of (dist_boundary, dist_closest/2)
        # If boundary is the limiting factor, move towards center.
        # If neighbor is the limiting factor, move away from neighbor.
        
        # Heuristic: move away from the closest entity (neighbor or boundary projection)
        # If boundary is closer than half the distance to neighbor, it's the bottleneck?
        # Actually, the score is min(dist_closest/2, dist_boundary).
        # Let's just pick the closer "constraint" in terms of the score contribution.
        
        # If dist_closest / 2 < dist_boundary, neighbor is limiting. Move away from neighbor.
        # Else, boundary is limiting. Move towards center.
        
        if dist_closest / 2.0 < dist_boundary:
            # Move away from closest neighbor
            direction = points[idx] - points[closest_idx]
            norm_dir = np.linalg.norm(direction)
            if norm_dir > 1e-9:
                direction = direction / norm_dir
            else:
                direction = np.array([1.0, 0.0])
            new_points = points.copy()
            new_points[idx] = points[idx] + direction * step_size
        else:
            # Move towards center (away from boundary)
            direction = -points[idx]
            norm_dir = np.linalg.norm(direction)
            if norm_dir > 1e-9:
                direction = direction / norm_dir
            else:
                # If at center, pick random direction
                angle = np.random.rand() * 2 * np.pi
                direction = np.array([np.cos(angle), np.sin(angle)])
            new_points = points.copy()
            new_points[idx] = points[idx] + direction * step_size
            
        # Ensure new point is not wildly outside (optional, score will handle it)
        # But if it goes way outside, score becomes negative, which is bad for SA.
        # Let's clip the new point to the unit circle? No, let's allow it but score will be low.
        # Actually, if a point is outside, min_boundary is negative. Score is negative.
        # We want to maximize score.
        
        new_score = get_score(new_points)
        
        # Acceptance criterion
        delta_score = new_score - current_score
        if delta_score > 0 or np.random.rand() < np.exp(delta_score / T):
            points = new_points
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_points = points.copy()
        
        # Cool down
        T *= alpha
        if T < T_min:
            T = T_min
            
        # Adaptive step size: if we are stuck (no improvement for a while), reduce step?
        # Or increase T?
        # Simple: just let T cool.
        
    # Final check
    # Ensure the returned points are valid (inside circle)
    radii = np.linalg.norm(best_points, axis=1)
    mask = radii > 1.0
    if np.any(mask):
        best_points[mask] = best_points[mask] / radii[mask][:, np.newaxis] * 0.999
        
    return best_points

# EVOLVE-BLOCK-END
