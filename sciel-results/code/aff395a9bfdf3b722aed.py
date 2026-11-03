import numpy as np
from scipy.optimize import minimize

N = 14

# Define fixed non-uniform angles clustering near diagonals
# Diagonals are at pi/4, 3pi/4, 5pi/4, 7pi/4
base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
# Add perturbation to cluster near diagonals
# sin(4*theta) is zero at diagonals, derivative cos(4*theta) is -1 at diagonals
# So theta + A*sin(4*theta) has smaller derivative near diagonals -> higher density
A = 0.15
angles = base_angles + A * np.sin(4 * base_angles)
angles = np.sort(angles)

def points_from_support(angles, h):
    """
    Compute vertices of convex polygon from support function values h at given angles.
    Vertex i is intersection of lines defined by (theta_{i-1}, h_{i-1}) and (theta_i, h_i).
    """
    n = len(angles)
    points = np.zeros((n, 2))
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)
    
    for i in range(n):
        i_prev = (i - 1) % n
        # Line i-1: x*cos(a_prev) + y*sin(a_prev) = h_prev
        # Line i:   x*cos(a_curr) + y*sin(a_curr) = h_curr
        a1, a2 = angles[i_prev], angles[i]
        h1, h2 = h[i_prev], h[i]
        
        c1, s1 = cos_angles[i_prev], sin_angles[i_prev]
        c2, s2 = cos_angles[i], sin_angles[i]
        
        det = c1 * s2 - c2 * s1
        if abs(det) < 1e-12:
            # Degenerate, shouldn't happen with distinct angles
            points[i] = [0, 0]
            continue
            
        # x = (h1*s2 - h2*s1) / det
        # y = (h2*c1 - h1*c2) / det
        points[i, 0] = (h1 * s2 - h2 * s1) / det
        points[i, 1] = (h2 * c1 - h1 * c2) / det
        
    return points

def convex_hull_area(points):
    """Calculate area using shoelace formula (points in CCW order)."""
    n = len(points)
    area = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0

def min_triangle_area(points):
    """Calculate the minimum area of any triangle formed by 3 distinct points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
                area = abs(cross) * 0.5
                if area < min_area:
                    min_area = area
    return min_area

def objective(u):
    """Objective: minimize negative score. u are log-support-values."""
    h = np.exp(u)
    points = points_from_support(angles, h)
    
    # Check for degeneracy or numerical issues
    if np.any(~np.isfinite(points)):
        return 1e6
        
    area_hull = convex_hull_area(points)
    if area_hull < 1e-9:
        return 1e6
        
    area_min_tri = min_triangle_area(points)
    score = area_min_tri / area_hull
    return -score

def construct():
    best_score = -1e6
    best_points = None
    
    rng = np.random.default_rng(42)
    
    # Multi-start Nelder-Mead
    for start_idx in range(10):
        if start_idx == 0:
            u0 = np.zeros(N) # Circle
        else:
            # Random perturbation around circle
            u0 = rng.normal(0, 0.1, N)
            
        result = minimize(
            objective,
            u0,
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8}
        )
        
        final_u = result.x
        final_h = np.exp(final_u)
        final_points = points_from_support(angles, final_h)
        
        # Verify convexity and compute score
        # The support function construction guarantees convexity if h>0 and angles are sorted/distinct
        # But let's double check area
        area_hull = convex_hull_area(final_points)
        if area_hull > 1e-9:
            area_min_tri = min_triangle_area(final_points)
            score = area_min_tri / area_hull
            if score > best_score:
                best_score = score
                best_points = final_points
    
    if best_points is None:
        # Fallback to circle
        h0 = np.ones(N)
        best_points = points_from_support(angles, h0)
        
    return best_points
