# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import itertools
import time


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    
    # Precompute all index combinations for triangles
    indices = list(itertools.combinations(range(n), 3))
    idx_arrays = np.array(indices) # shape (286, 3)
    
    def evaluate(points):
        """
        Vectorized evaluation of the metric: min_triangle_area / hull_area.
        Returns 0 if invalid.
        """
        if not np.all(np.isfinite(points)):
            return 0.0
            
        # Compute hull area
        try:
            hull = ConvexHull(points)
            a_hull = hull.volume
        except Exception:
            return 0.0
            
        if a_hull < 1e-10:
            return 0.0
            
        # Compute all triangle areas vectorized
        # points: (13, 2)
        # idx_arrays: (286, 3)
        p1 = points[idx_arrays[:, 0]]
        p2 = points[idx_arrays[:, 1]]
        p3 = points[idx_arrays[:, 2]]
        
        # Area = 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
        # = 0.5 * | (x2-x1)(y3-y1) - (x3-x1)(y2-y1) |
        # Using cross product magnitude in 2D
        v1 = p2 - p1
        v2 = p3 - p1
        areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
        
        a_min = np.min(areas)
        
        return a_min / a_hull

    def optimize():
        # Initial guess: Regular polygon is okay, but let's try a spiral or grid
        # A good heuristic for Heilbronn is often related to lattice points or well-spaced points.
        # Let's use a random start within a square, then optimize.
        
        rng = np.random.default_rng(42)
        
        # Try a few different initial configurations and pick the best start for SA
        # 1. Regular polygon
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        p_poly = np.column_stack([np.cos(angles), np.sin(angles)])
        
        # 2. Grid-like (sqrt(13) x sqrt(13) approx)
        # 13 points: 3x5 grid with adjustments? Or just random.
        p_grid = rng.uniform(-1, 1, (n, 2))
        
        # Evaluate starts
        score_poly = evaluate(p_poly)
        score_grid = evaluate(p_grid)
        
        # Start with the better one
        if score_poly > score_grid:
            current_points = p_poly.copy()
            current_score = score_poly
        else:
            current_points = p_grid.copy()
            current_score = score_grid
            
        best_points = current_points.copy()
        best_score = current_score
        
        # Simulated Annealing
        T = 1e-4
        T_min = 1e-10
        alpha = 0.9995
        max_iter = 20000
        step_size = 0.1
        
        start_time = time.time()
        
        for i in range(max_iter):
            if time.time() - start_time > 500: # Leave 100s buffer
                break
                
            # Generate new point set by perturbing current
            new_points = current_points + rng.normal(0, step_size, (n, 2))
            
            new_score = evaluate(new_points)
            
            if new_score > current_score:
                current_points = new_points
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_points = new_points.copy()
            else:
                # Accept worse solution with probability exp((new - current) / T)
                # Since we maximize, delta = new - current < 0
                # prob = exp(delta / T)
                delta = new_score - current_score
                if delta < 0:
                    prob = np.exp(delta / T)
                    if rng.random() < prob:
                        current_points = new_points
                        current_score = new_score
            
            T *= alpha
            if T < T_min:
                T = T_min
                
            # Adaptive step size? 
            # If stuck, maybe increase step? Or just rely on T.
            # Let's slowly decrease step size too
            if i % 1000 == 0:
                step_size *= 0.9

        return best_points

    points = optimize()
    
    # Final polish with a local optimizer if possible? 
    # SA might be enough. Let's ensure validity.
    if not np.all(np.isfinite(points)):
        # Fallback
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return points


# EVOLVE-BLOCK-END
