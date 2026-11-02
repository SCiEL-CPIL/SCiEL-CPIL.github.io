"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import least_squares
from itertools import combinations

def _convex_hull_area(points):
    """Compute the area of the convex hull of a set of 2D points."""
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _triangle_area(p1, p2, p3):
    """Compute area of triangle formed by three points."""
    return 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))

def _all_triangle_areas(points):
    """Compute all triangle areas."""
    n = len(points)
    areas = []
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                area = _triangle_area(points[i], points[j], points[k])
                areas.append(area)
                indices.append((i, j, k))
    return np.array(areas), indices

def _residuals(x, active_indices, target_A):
    """
    Residuals for the equi-area system.
    x: flattened point coordinates (2N,)
    active_indices: list of (i,j,k) tuples for active triangles
    target_A: target area for active triangles
    
    Returns residual vector:
    - For each active triangle: area - target_A
    - Normalization: (hull_area - 1)
    """
    points = x.reshape(-1, 2)
    residuals = []
    
    # Area residuals
    for (i, j, k) in active_indices:
        area = _triangle_area(points[i], points[j], points[k])
        residuals.append(area - target_A)
    
    # Hull area normalization
    hull_area = _convex_hull_area(points)
    residuals.append(hull_area - 1.0)
    
    return np.array(residuals)

def construct():
    """
    Place 13 points to maximize min triangle area / convex hull area.
    Uses equi-area least-squares approach based on the theoretical property
    that optimal Heilbronn configurations have equal-area minimum triangles.
    """
    # Start with regular 13-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Normalize to unit hull area
    hull_area = _convex_hull_area(points)
    if hull_area > 1e-10:
        points = points / np.sqrt(hull_area)
    
    areas, _ = _all_triangle_areas(points)
    best_score = np.min(areas) / _convex_hull_area(points)
    best_points = points.copy()
    
    # Iteratively improve using equi-area least squares
    for outer_iter in range(15):
        areas, indices = _all_triangle_areas(points)
        min_area = np.min(areas)
        
        if min_area < 1e-15:
            break
        
        # Find active triangles: those within 15% of minimum
        active_mask = areas <= (min_area + 0.15 * min_area)
        active_indices = [indices[idx] for idx, mask in enumerate(active_mask) if mask]
        
        # Limit to avoid over-determined system
        max_active = min(len(active_indices), 2*N)
        if len(active_indices) > max_active:
            # Keep the smallest area triangles
            sorted_active = sorted(enumerate(active_indices), key=lambda x: areas[x[0]])
            active_indices = [x[1] for x in sorted_active[:max_active]]
        
        if len(active_indices) < 2:
            break
        
        # Solve using least_squares
        try:
            result = least_squares(
                _residuals,
                points.flatten(),
                args=(active_indices, min_area),
                method='lm',
                max_nfev=1000,
                xtol=1e-12,
                ftol=1e-12
            )
            
            if result.success or result.cost < 1e-10:
                new_points = result.x.reshape(-1, 2)
                
                # Check if this is valid and improved
                new_hull = _convex_hull_area(new_points)
                if new_hull > 1e-10:
                    new_areas, _ = _all_triangle_areas(new_points)
                    new_min = np.min(new_areas)
                    new_score = new_min / new_hull
                    
                    if new_score > best_score:
                        best_score = new_score
                        best_points = new_points.copy()
                        points = new_points.copy()
        except Exception:
            pass
        
        # Normalize
        hull_area = _convex_hull_area(points)
        if hull_area > 1e-10:
            points = points / np.sqrt(hull_area)
    
    # Return the best points found
    hull_area = _convex_hull_area(best_points)
    if hull_area > 1e-10:
        best_points = best_points / np.sqrt(hull_area)
    
    return best_points
# EVOLVE-BLOCK-END
