"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize_scalar

def _score(pts):
    """
    Compute the score: min triangle area / convex hull area.
    Returns 0 if points are collinear or hull area is 0.
    """
    if len(pts) < 3:
        return 0.0
    
    # Calculate Convex Hull Area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume # In 2D, volume is area
    except Exception:
        return 0.0
        
    if hull_area < 1e-12:
        return 0.0
        
    # Calculate Minimum Triangle Area
    # Vectorized calculation for all triples
    n = len(pts)
    min_area = float('inf')
    
    # Efficient vectorized computation
    # We iterate over pairs (i, j) and vectorize over k
    for i in range(n - 2):
        pi = pts[i]
        for j in range(i + 1, n - 1):
            pj = pts[j]
            pk = pts[j+1:]
            
            # Cross product of (pj - pi) and (pk - pi)
            v1x = pj[0] - pi[0]
            v1y = pj[1] - pi[1]
            v2x = pk[:, 0] - pi[0]
            v2y = pk[:, 1] - pi[1]
            
            crosses = np.abs(v1x * v2y - v1y * v2x)
            areas = 0.5 * crosses
            
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
                
    return min_area / hull_area

def _generate_two_ring_points(r, theta):
    """
    Generate 14 points: 7 on outer ring (radius 1), 7 on inner ring (radius r).
    Outer angles: 0, 2pi/7, ...
    Inner angles: theta, theta + 2pi/7, ...
    """
    angles_outer = np.linspace(0, 2*np.pi, 7, endpoint=False)
    outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    
    angles_inner = np.linspace(theta, theta + 2*np.pi, 7, endpoint=False)
    inner_pts = np.column_stack([r * np.cos(angles_inner), r * np.sin(angles_inner)])
    
    pts = np.vstack([outer_pts, inner_pts])
    return pts

def _fast_analyze(r, theta):
    """
    Fast analysis of the 2-ring configuration.
    Returns (min_triangle_area, hull_area).
    """
    pts = _generate_two_ring_points(r, theta)
    
    # Convex hull area: since outer points are on unit circle and form a convex polygon,
    # and inner points are inside, the convex hull is the outer heptagon.
    # Area of regular heptagon with circumradius R=1:
    # A = (n/2) * R^2 * sin(2*pi/n)
    n_outer = 7
    hull_area = (n_outer / 2.0) * 1.0 * np.sin(2 * np.pi / n_outer)
    
    # Calculate min triangle area
    # We can vectorize this significantly.
    # There are 3 types of triangles:
    # 1. 3 outer points
    # 2. 2 outer, 1 inner
    # 3. 1 outer, 2 inner
    
    # 1. 3 outer points
    # The minimum area triangle from 3 vertices of a regular heptagon is formed by 3 consecutive vertices.
    # Area = 0.5 * | (P2-P1) x (P3-P1) |
    # Let's compute all outer triangle areas to be safe, though we know the min is consecutive.
    # Actually, for regular n-gon, min area is indeed consecutive.
    # Area of triangle formed by vertices 0, 1, 2 of regular heptagon (R=1):
    # Side length s = 2 * sin(pi/7)
    # Height h = 1 - cos(2*pi/7) ? No.
    # Let's just compute it numerically for robustness or use formula.
    # Formula: Area = 0.5 * R^2 * (sin(2pi/n) + sin(2pi/n) - sin(4pi/n)) ? 
    # Let's just compute the min of outer triangles directly.
    outer_pts = pts[:7]
    min_outer = float('inf')
    for i in range(7):
        for j in range(i+1, 7):
            for k in range(j+1, 7):
                p1, p2, p3 = outer_pts[i], outer_pts[j], outer_pts[k]
                area = 0.5 * np.abs(np.cross(p2-p1, p3-p1))
                if area < min_outer:
                    min_outer = area
    
    # 2. 2 outer, 1 inner
    # 3. 1 outer, 2 inner
    
    # We can vectorize the calculation for all pairs of outer points and all inner points, etc.
    # But 7x7x7 is small enough that a simple loop or partial vectorization is fine.
    # Let's vectorize the "2 outer, 1 inner" and "1 outer, 2 inner" parts.
    
    inner_pts = pts[7:]
    
    # 2 outer, 1 inner
    # For each pair of outer points (i, j), and each inner point k, compute area.
    # There are 21 pairs of outer points and 7 inner points. 147 triangles.
    # We can do this with broadcasting.
    
    # Get all outer pairs
    outer_indices = np.arange(7)
    oi, oj = np.meshgrid(outer_indices, outer_indices, indexing='ij')
    oi_flat = oi.ravel()
    oj_flat = oj.ravel()
    
    # Keep only i < j
    mask = oi_flat < oj_flat
    oi_flat = oi_flat[mask]
    oj_flat = oj_flat[mask]
    
    # Pairs: (21, 2)
    p1 = outer_pts[oi_flat]
    p2 = outer_pts[oj_flat]
    
    # Inner points: (7, 2)
    p3 = inner_pts
    
    # Compute areas: 0.5 * | (p2 - p1) x (p3 - p1) |
    # We need to broadcast p1, p2 (21, 2) with p3 (7, 2).
    # p1_expanded: (21, 1, 2)
    # p2_expanded: (21, 1, 2)
    # p3_expanded: (1, 7, 2)
    
    p1_e = p1[:, np.newaxis, :]
    p2_e = p2[:, np.newaxis, :]
    p3_e = p3[np.newaxis, :, :]
    
    v1 = p2_e - p1_e
    v2 = p3_e - p1_e
    
    cross_x = v1[:, :, 0] * v2[:, :, 1] - v1[:, :, 1] * v2[:, :, 0]
    areas_2o1i = 0.5 * np.abs(cross_x)
    min_2o1i = np.min(areas_2o1i)
    
    # 1 outer, 2 inner
    # For each outer point i, and each pair of inner points (j, k).
    # 7 outer points, 21 inner pairs. 147 triangles.
    
    inner_indices = np.arange(7)
    ii, ij = np.meshgrid(inner_indices, inner_indices, indexing='ij')
    ii_flat = ii.ravel()
    ij_flat = ij.ravel()
    
    mask = ii_flat < ij_flat
    ii_flat = ii_flat[mask]
    ij_flat = ij_flat[mask]
    
    p2 = inner_pts[ii_flat]
    p3 = inner_pts[ij_flat]
    
    p1 = outer_pts
    
    p1_e = p1[:, np.newaxis, :]
    p2_e = p2[np.newaxis, :, :]
    p3_e = p3[np.newaxis, :, :]
    
    v1 = p2_e - p1_e
    v2 = p3_e - p1_e
    
    cross_x = v1[:, :, 0] * v2[:, :, 1] - v1[:, :, 1] * v2[:, :, 0]
    areas_1o2i = 0.5 * np.abs(cross_x)
    min_1o2i = np.min(areas_1o2i)
    
    min_area = min(min_outer, min_2o1i, min_1o2i)
    
    return min_area, hull_area

def _negative_score(params):
    """
    Objective function for optimization.
    params: array of shape (2,) representing [r, theta].
    """
    r, theta = params
    if r <= 0 or r >= 1:
        return 1e9
    if theta < 0 or theta >= 2*np.pi:
        # Wrap theta
        theta = theta % (2*np.pi)
        
    min_area, hull_area = _fast_analyze(r, theta)
    if hull_area < 1e-12:
        return 1e9
        
    return -(min_area / hull_area)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin for 120s limit
    
    # We will optimize over r and theta.
    # The problem is 2D.
    # We can use a grid search to find a good initial guess, then Nelder-Mead.
    
    best_params = None
    best_score = -np.inf
    
    # Grid search for initial guess
    # r in [0.2, 0.8], theta in [0, 2*pi/7] (symmetry period)
    # Let's try a coarse grid.
    r_vals = np.linspace(0.2, 0.8, 20)
    theta_vals = np.linspace(0, 2*np.pi/7, 20)
    
    for r in r_vals:
        if time.time() > deadline - 50:
            break
        for theta in theta_vals:
            if time.time() > deadline - 50:
                break
            score = -_negative_score([r, theta])
            if score > best_score:
                best_score = score
                best_params = [r, theta]
                
    # Refine with Nelder-Mead
    if best_params is not None:
        # Define a wrapper for minimize that handles the 2D input
        def obj_2d(params):
            return _negative_score(params)
            
        res = minimize(
            obj_2d,
            best_params,
            method='Nelder-Mead',
            options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 500, 'adaptive': True}
        )
        
        final_score = -res.fun
        if final_score > best_score:
            best_score = final_score
            best_params = res.x
            
    if best_params is None:
        # Fallback to rings if optimization failed
        pts = _generate_two_ring_points(0.5, np.pi/7)
        return pts

    r, theta = best_params
    pts = _generate_two_ring_points(r, theta)
    
    return pts
# EVOLVE-BLOCK-END
