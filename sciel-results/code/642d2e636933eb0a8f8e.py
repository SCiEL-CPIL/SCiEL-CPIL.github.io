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
    
    def triangle_areas(points):
        """Compute areas of all triangles formed by the points."""
        areas = []
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    p1, p2, p3 = points[i], points[j], points[k]
                    area = 0.5 * abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))
                    areas.append(area)
        return np.array(areas)

    def hull_area_and_indices(points):
        """Compute area of convex hull and indices of points on the hull."""
        try:
            hull = ConvexHull(points)
            area = hull.volume # In 2D, volume is area
            hull_indices = hull.simplices[:, 0]
            return area, hull_indices
        except:
            return 0, []

    def score(points):
        """Compute the ratio A_min / A_hull."""
        areas = triangle_areas(points)
        min_area = np.min(areas)
        hull_area, _ = hull_area_and_indices(points)
        if hull_area == 0:
            return 0
        return min_area / hull_area

    def gradient(points):
        """Compute gradient of the score function."""
        areas = triangle_areas(points)
        min_area = np.min(areas)
        hull_area, hull_indices = hull_area_and_indices(points)
        
        if hull_area == 0:
            return np.zeros_like(points)
            
        # Gradient of A_min
        # Find all triangles with area close to min_area
        # Use a small tolerance to handle ties
        tol = 1e-6
        min_indices = np.where(areas <= min_area + tol)[0]
        
        grad_min = np.zeros((n, 2))
        
        # Map triangle index to (i, j, k)
        # We need to reconstruct i, j, k from the linear index
        # Total triangles = n*(n-1)*(n-2)/6
        # Let's just re-iterate to find the gradients for the min triangles
        # This is inefficient but n=13 is small (286 triangles)
        
        # Precompute i, j, k for all triangles
        tri_indices = []
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    tri_indices.append((i, j, k))
                    idx += 1
        
        for m in min_indices:
            i, j, k = tri_indices[m]
            p1, p2, p3 = points[i], points[j], points[k]
            # Area = 0.5 * |cross(p2-p1, p3-p1)|
            # Let cross = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
            # Area = 0.5 * |cross|
            # Gradient of cross w.r.t p1, p2, p3
            # cross = (x2-x1)(y3-y1) - (y2-y1)(x3-x1)
            # d/dx1: -(y3-y1) - (y2-y1)*(0 - 1) = -y3+y1 + y2-y1 = y2-y3
            # d/dy1: -(x2-x1) - (0) = -x2+x1 = x1-x2
            # d/dx2: (y3-y1) - 0 = y3-y1
            # d/dy2: 0 - (x3-x1) = x1-x3
            # d/dx3: 0 - (y2-y1) = y1-y2
            # d/dy3: (x2-x1) - 0 = x2-x1
            
            # Note: Area = 0.5 * |cross|. 
            # If cross > 0, grad Area = 0.5 * grad cross.
            # If cross < 0, grad Area = -0.5 * grad cross.
            cross = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0])
            sign = 1.0 if cross > 0 else -1.0
            
            g1 = 0.5 * sign * np.array([p2[1]-p3[1], p1[0]-p2[0]])
            g2 = 0.5 * sign * np.array([p3[1]-p1[1], p1[0]-p3[0]])
            g3 = 0.5 * sign * np.array([p1[1]-p2[1], p2[0]-p1[0]])
            
            grad_min[i] += g1
            grad_min[j] += g2
            grad_min[k] += g3

        # Normalize gradient of min area if multiple min triangles?
        # The subgradient is the sum. This is fine.
        
        # Gradient of Hull Area
        grad_hull = np.zeros((n, 2))
        if len(hull_indices) > 0:
            # Sort hull indices to get the order
            # ConvexHull.simplices gives triangles, not the ordered boundary directly in a simple list
            # But hull.vertices gives the vertices in CCW order
            hull_vertices = hull.vertices
            m = len(hull_vertices)
            for idx_h in range(m):
                v_curr = hull_vertices[idx_h]
                v_prev = hull_vertices[(idx_h - 1) % m]
                v_next = hull_vertices[(idx_h + 1) % m]
                
                p_curr = points[v_curr]
                p_prev = points[v_prev]
                p_next = points[v_next]
                
                # Area contribution of triangle (origin, p_prev, p_curr) + (origin, p_curr, p_next) ... 
                # Actually, A_hull = 0.5 * sum (x_i y_{i+1} - x_{i+1} y_i)
                # Term involving p_curr:
                # From (v_prev, v_curr): 0.5 * (x_prev y_curr - x_curr y_prev)
                # From (v_curr, v_next): 0.5 * (x_curr y_next - x_next y_curr)
                
                # d/dx_curr: 0.5 * (-y_prev + y_next)
                # d/dy_curr: 0.5 * (x_prev - x_next)
                
                grad_hull[v_curr][0] += 0.5 * (-p_prev[1] + p_next[1])
                grad_hull[v_curr][1] += 0.5 * (p_prev[0] - p_next[0])

        # Gradient of Ratio R = A_min / A_hull
        # dR = (A_hull * dA_min - A_min * dA_hull) / A_hull^2
        grad_ratio = (hull_area * grad_min - min_area * grad_hull) / (hull_area * hull_area)
        
        return grad_ratio

    def optimize(points, max_iters=2000, step_size=0.05):
        """Perform gradient ascent on the score."""
        best_points = points.copy()
        best_score = score(points)
        
        for i in range(max_iters):
            grad = gradient(points)
            # Normalize gradient to avoid huge steps
            norm = np.linalg.norm(grad)
            if norm > 1e-10:
                grad = grad / norm
            
            # Update
            new_points = points + step_size * grad
            
            # Check validity
            if not np.all(np.isfinite(new_points)):
                break
            
            new_score = score(new_points)
            
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
                points = new_points.copy()
            else:
                # If no improvement, try smaller step or break
                step_size *= 0.9
                if step_size < 1e-5:
                    break
                    
        return best_points, best_score

    # Initial Configuration: Triangular Lattice
    # Generate a 4x4 triangular lattice (16 points) and pick 13
    # Points: (i + j/2, j * sqrt(3)/2) for i in 0..3, j in 0..3
    # We want to pick 13 points.
    # Let's try removing the 3 corners that are "least central" or just take the first 13.
    # A better start: Center of mass at origin.
    
    lattice_points = []
    for j in range(4):
        for i in range(4):
            x = i + j/2.0
            y = j * np.sqrt(3)/2.0
            lattice_points.append([x, y])
            
    lattice_points = np.array(lattice_points)
    
    # Select 13 points.
    # The 16 points form a rhombus.
    # Removing 3 points. Which ones?
    # Let's try removing the 3 "outermost" points that are not critical for the min triangle?
    # Or just take a subset that is roughly symmetric.
    # Indices:
    # (0,0) (1,0) (2,0) (3,0)
    # (0,1) (1,1) (2,1) (3,1)
    # (0,2) (1,2) (2,2) (3,2)
    # (0,3) (1,3) (2,3) (3,3)
    
    # Let's remove (3,0), (3,1), (3,2)? No, that breaks symmetry.
    # Let's remove (3,3), (0,3), (3,0)?
    # Let's just try a few subsets.
    
    subsets = [
        list(range(13)), # First 13
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        [0, 1, 2, 5, 6, 7, 8, 9, 10, 13, 14, 15, 12], # Some mix
        [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15]
    ]
    
    best_global_points = None
    best_global_score = 0
    
    start_time = time.time()
    deadline = start_time + 580 # 20s margin
    
    for subset in subsets:
        if time.time() > deadline:
            break
            
        points = lattice_points[subset].copy()
        
        # Center the points
        points -= np.mean(points, axis=0)
        
        # Optimize
        opt_points, opt_score = optimize(points, max_iters=1000, step_size=0.1)
        
        if opt_score > best_global_score:
            best_global_score = opt_score
            best_global_points = opt_points.copy()
            
    # If time remains, do a final polish with smaller steps
    if time.time() < deadline and best_global_points is not None:
        opt_points, opt_score = optimize(best_global_points, max_iters=2000, step_size=0.01)
        if opt_score > best_global_score:
            best_global_points = opt_points.copy()
            best_global_score = opt_score

    if best_global_points is None:
        # Fallback to random
        rng = np.random.default_rng(seed=42)
        best_global_points = rng.random((n, 2))
        
    return best_global_points

# EVOLVE-BLOCK-END
