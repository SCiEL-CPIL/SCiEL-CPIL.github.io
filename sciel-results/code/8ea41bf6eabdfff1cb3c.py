import numpy as np
import time
from scipy.spatial.distance import pdist

def _score(points):
    """Compute (d_min / d_max)^2 for a set of points."""
    if points.shape != (14, 3):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    dists = pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0:
        return 0.0
    return (d_min / d_max) ** 2

def _fibonacci_sphere(n):
    points = []
    golden_ratio = (1 + np.sqrt(5)) / 2
    for i in range(n):
        z = 1 - 2 * (i + 0.5) / n
        r = np.sqrt(1 - z**2)
        theta = 2 * np.pi * i / golden_ratio
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points.append([x, y, z])
    return np.array(points)

def _min_dist(points):
    """Return the minimum pairwise distance."""
    dists = pdist(points)
    return np.min(dists)

def _project_to_sphere(points):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1e-12
    return points / norms

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize (d_min / d_max)^2.
    Uses TSP-style 2-opt and 3-opt local search on the unit sphere.
    """
    start_time = time.time()
    deadline = start_time + 620  # Safety margin before 630s kill
    
    n = 14
    best_points = _fibonacci_sphere(n)
    best_score = _score(best_points)
    
    rng = np.random.default_rng(42)
    
    # Run multiple independent search attempts from different starts
    num_starts = 12
    
    for start_idx in range(num_starts):
        if time.time() > deadline - 10:
            break
        
        # Different initializations
        if start_idx == 0:
            points = _fibonacci_sphere(n)
        elif start_idx == 1:
            # Random on sphere
            points = rng.standard_normal((n, 3))
            points = _project_to_sphere(points)
        elif start_idx == 2:
            # Slightly perturbed fibonacci
            points = _fibonacci_sphere(n) + rng.standard_normal((n, 3)) * 0.1
            points = _project_to_sphere(points)
        elif start_idx == 3:
            # Two antipodal clusters
            half = n // 2
            pts1 = rng.standard_normal((half, 3))
            pts2 = -pts1
            points = np.vstack([pts1, pts2])
            if n % 2 != 0:
                points = np.vstack([points, rng.standard_normal((1, 3))])
            points = _project_to_sphere(points)
        else:
            # Random with increasing perturbation
            points = _fibonacci_sphere(n) + rng.standard_normal((n, 3)) * (0.05 + 0.05 * start_idx)
            points = _project_to_sphere(points)
        
        # Local search: 2-opt and 3-opt moves
        # We maximize d_min, so we accept moves that increase the minimum distance
        current_min = _min_dist(points)
        
        # Phase 1: 2-opt (swap two points)
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(n):
                if time.time() > deadline - 5:
                    break
                for j in range(i + 1, n):
                    # Try swapping points i and j
                    # But swapping identical points doesn't help; instead, 
                    # we try to move point i toward or away from others.
                    # Actually, for sphere packing, a better move is to 
                    # reflect or rotate. Let's use a different strategy:
                    # Try to move point i to the position that maximizes 
                    # its minimum distance to all other points.
                    # This is a 1-opt move: reposition one point.
                    pass
        
        # Better approach: 1-opt repositioning
        # For each point, try to find a better position on the sphere
        # by sampling random directions and keeping the best
        improved = True
        max_rounds = 50
        round_num = 0
        while improved and round_num < max_rounds and time.time() < deadline - 5:
            improved = False
            round_num += 1
            
            for i in range(n):
                if time.time() > deadline - 5:
                    break
                
                # Try to improve point i
                best_pos = points[i]
                best_min_dist = current_min
                
                # Sample candidate positions
                # 1. Random directions
                for _ in range(20):
                    candidate = rng.standard_normal(3)
                    candidate /= np.linalg.norm(candidate)
                    
                    # Place candidate at position i
                    new_points = points.copy()
                    new_points[i] = candidate
                    
                    # Compute new minimum distance
                    # Only need to check distances involving point i
                    # and the overall minimum
                    new_dists = pdist(new_points)
                    new_min = np.min(new_dists)
                    
                    if new_min > best_min_dist + 1e-10:
                        best_min_dist = new_min
                        best_pos = candidate
                        improved = True
                
                # 2. Move away from nearest neighbor
                # Find nearest neighbor of point i
                dists_i = np.linalg.norm(points - points[i], axis=1)
                dists_i[i] = np.inf
                nearest = np.argmin(dists_i)
                
                # Move in direction away from nearest
                direction = points[i] - points[nearest]
                direction /= np.linalg.norm(direction)
                
                # Try several steps in this direction
                for step in [0.1, 0.2, 0.3, 0.5]:
                    candidate = points[i] + step * direction
                    candidate = _project_to_sphere(candidate.reshape(1, 3))[0]
                    
                    new_points = points.copy()
                    new_points[i] = candidate
                    new_min = np.min(pdist(new_points))
                    
                    if new_min > best_min_dist + 1e-10:
                        best_min_dist = new_min
                        best_pos = candidate
                        improved = True
                
                # 3. Move toward the "hole" - direction of maximum minimum distance
                # Approximate by averaging directions away from all neighbors
                away_dirs = points[i] - points
                away_dirs[i] = 0
                away_dir = np.sum(away_dirs, axis=0)
                if np.linalg.norm(away_dir) > 1e-10:
                    away_dir /= np.linalg.norm(away_dir)
                    candidate = points[i] + 0.2 * away_dir
                    candidate = _project_to_sphere(candidate.reshape(1, 3))[0]
                    
                    new_points = points.copy()
                    new_points[i] = candidate
                    new_min = np.min(pdist(new_points))
                    
                    if new_min > best_min_dist + 1e-10:
                        best_min_dist = new_min
                        best_pos = candidate
                        improved = True
                
                points[i] = best_pos
                if best_min_dist > current_min:
                    current_min = best_min_dist
        
        # Check if this configuration is better
        points = _project_to_sphere(points)
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
    
    return best_points
