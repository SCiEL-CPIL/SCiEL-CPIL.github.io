# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time
import itertools

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points on or inside a convex region to maximize the
    ratio A_min / A_hull, where A_min is the minimum triangle area and A_hull is the
    convex hull area.
    
    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    n = 13
    
    def triangle_areas(points):
        """Compute all triangle areas from a set of points."""
        n_pts = len(points)
        areas = np.empty((n_pts, n_pts, n_pts))
        for i in range(n_pts):
            for j in range(i+1, n_pts):
                for k in range(j+1, n_pts):
                    p1, p2, p3 = points[i], points[j], points[k]
                    area = abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1])) / 2.0
                    areas[i,j,k] = area
                    areas[j,i,k] = area
                    areas[i,k,j] = area
                    areas[k,j,i] = area
                    areas[j,k,i] = area
                    areas[k,i,j] = area
        return areas
    
    def min_triangle_area(points):
        """Compute minimum triangle area among all triples."""
        n_pts = len(points)
        min_area = float('inf')
        for i in range(n_pts):
            for j in range(i+1, n_pts):
                for k in range(j+1, n_pts):
                    p1, p2, p3 = points[i], points[j], points[k]
                    area = abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1])) / 2.0
                    if area < min_area:
                        min_area = area
        return min_area
    
    def hull_area(points):
        """Compute convex hull area."""
        try:
            hull = ConvexHull(points)
            return hull.volume  # In 2D, volume is area
        except Exception:
            return float('inf')
    
    def score(points):
        """Compute the score: A_min / A_hull."""
        a_min = min_triangle_area(points)
        a_hull = hull_area(points)
        if a_hull <= 0 or not np.isfinite(a_hull):
            return 0.0
        if not np.isfinite(a_min):
            return 0.0
        return a_min / a_hull
    
    def normalize(points):
        """Normalize points to unit area convex hull for better optimization."""
        try:
            hull = ConvexHull(points)
            area = hull.volume
            if area <= 0:
                return points
            # Center at centroid
            centroid = points.mean(axis=0)
            points_centered = points - centroid
            # Scale to unit area
            points_scaled = points_centered / np.sqrt(area)
            return points_scaled
        except Exception:
            return points
    
    def generate_initial_points(method='grid', seed=42):
        """Generate initial point configurations."""
        rng = np.random.default_rng(seed)
        
        if method == 'grid':
            # Start with a structured grid-like arrangement
            # 13 points: maybe 4x4 grid minus one corner, or other patterns
            # Try a hexagonal-like pattern
            points = np.array([
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, np.sqrt(3)/2],
                [1.5, np.sqrt(3)/2],
                [2.0, 0.0],
                [0.5, np.sqrt(3)],
                [1.5, np.sqrt(3)],
                [2.5, np.sqrt(3)/2],
                [3.0, 0.0],
                [1.0, np.sqrt(3)*1.5],
                [2.0, np.sqrt(3)*1.5],
                [3.0, np.sqrt(3)*1.5],
                [2.0, np.sqrt(3)*2],
            ])
            # Add small random perturbation
            points += rng.random((n, 2)) * 0.01
            return points
        
        elif method == 'random':
            return rng.random((n, 2)) * 2 - 1
        
        elif method == 'circle':
            # Points on a circle
            angles = np.linspace(0, 2*np.pi, n, endpoint=False)
            points = np.column_stack([np.cos(angles), np.sin(angles)])
            points += rng.random((n, 2)) * 0.01
            return points
        
        else:
            return rng.random((n, 2))
    
    def optimize_local(points, max_iterations=2000, learning_rate=0.01):
        """Local optimization using gradient-free method (Nelder-Mead style)."""
        current_points = points.copy()
        current_score = score(current_points)
        
        best_points = current_points.copy()
        best_score = current_score
        
        # Simple hill climbing with random perturbations
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            
            # Random perturbation
            perturbation = np.random.default_rng(iteration).normal(0, learning_rate, (n, 2))
            new_points = current_points + perturbation
            
            new_score = score(new_points)
            
            if new_score > current_score:
                current_points = new_points
                current_score = new_score
                if current_score > best_score:
                    best_points = current_points.copy()
                    best_score = current_score
            else:
                # Reduce learning rate occasionally
                if iteration % 100 == 0:
                    learning_rate *= 0.95
        
        return best_points, best_score
    
    def optimize_with_scipy(points, maxiter=1000):
        """Use scipy.optimize.minimize to maximize the score."""
        from scipy.optimize import minimize
        
        def negative_score(flat_points):
            pts = flat_points.reshape(n, 2)
            if not np.all(np.isfinite(pts)):
                return 0.0
            return -score(pts)
        
        try:
            result = minimize(
                negative_score,
                points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': maxiter, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if result.success and result.fun < 0:
                return result.x.reshape(n, 2), -result.fun
            else:
                return points, score(points)
        except Exception:
            return points, score(points)
    
    # Try multiple initial configurations
    best_points = None
    best_score = -1
    
    methods = ['grid', 'random', 'circle']
    seeds = [42, 123, 456, 789]
    
    for method in methods:
        for seed in seeds:
            if time.time() > deadline:
                break
            
            points = generate_initial_points(method, seed)
            
            # First try local optimization
            opt_points, opt_score = optimize_local(points, max_iterations=500, learning_rate=0.05)
            
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_points
            
            # Then try scipy optimization if time permits
            if time.time() < deadline - 60:
                scipy_points, scipy_score = optimize_with_scipy(opt_points, maxiter=500)
                if scipy_score > best_score:
                    best_score = scipy_score
                    best_points = scipy_points
        
        if time.time() > deadline - 30:
            break
    
    # Final refinement if time permits
    if best_points is not None and time.time() < deadline - 10:
        final_points, final_score = optimize_local(best_points, max_iterations=1000, learning_rate=0.01)
        if final_score > best_score:
            best_points = final_points
            best_score = final_score
    
    if best_points is None:
        # Fallback: return random points
        rng = np.random.default_rng(42)
        best_points = rng.random((n, 2))
    
    return best_points

# EVOLVE-BLOCK-END
