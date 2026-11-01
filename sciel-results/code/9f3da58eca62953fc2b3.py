"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import itertools

def _score(points):
    """Calculate the score: min triangle area / convex hull area."""
    n = len(points)
    # Calculate convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume # In 2D, volume is area
        if hull_area < 1e-10:
            return 0.0
    except Exception:
        return 0.0
    
    # Calculate minimum triangle area
    min_area = float('inf')
    # Iterate over all triples
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                # Area of triangle = 0.5 * |cross(p2-p1, p3-p1)|
                cross = abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0]))
                area = 0.5 * cross
                if area < min_area:
                    min_area = area
                    
    return min_area / hull_area

def construct():
    # Stochastic Local Search (Hill Climbing)
    # Initialize with random points in the unit circle
    best_points = None
    best_score = 0.0
    
    # Number of restarts to increase chance of finding a good local optimum
    restarts = 10
    iterations_per_restart = 5000
    
    for r in range(restarts):
        # Initialize
        points = np.random.uniform(-1, 1, (N, 2))
        # Ensure they are not all in a tiny cluster
        # Normalize to unit circle roughly?
        # Actually, scale invariance means we can just work with them.
        # But keeping them in [-1,1] is a good prior.
        
        current_score = _score(points)
        
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
            
        # Hill climbing
        for i in range(iterations_per_restart):
            # Pick a random point to move
            idx = np.random.randint(N)
            
            # Generate a new position
            # Small Gaussian perturbation
            step_size = 0.1
            new_point = points[idx] + np.random.normal(0, step_size, 2)
            
            # Optionally, keep within a bounding box to prevent drift
            # If we don't constrain, the points might drift to infinity, 
            # but the score is scale invariant. However, numerical issues might arise.
            # Let's constrain to [-2, 2] to be safe.
            new_point = np.clip(new_point, -2, 2)
            
            # Update point
            old_point = points[idx].copy()
            points[idx] = new_point
            
            new_score = _score(points)
            
            if new_score > current_score:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
            else:
                # Revert
                points[idx] = old_point
                
    return best_points
# EVOLVE-BLOCK-END
