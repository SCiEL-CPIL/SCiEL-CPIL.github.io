"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull
from itertools import combinations
import time

def _min_triangle_area(points):
    """Compute minimum triangle area over all triples."""
    n = len(points)
    min_area = np.inf
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                v1 = points[j] - points[i]
                v2 = points[k] - points[i]
                cross = v1[0]*v2[1] - v1[1]*v2[0]
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0

def _score(points):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _lines_to_polygon(angles, offsets, n_lines=13):
    """
    Convert n support lines (angle, offset) to a convex polygon.
    Each line: x*cos(a) + y*sin(a) = offset
    The polygon is the intersection of all half-planes x*cos(a) + y*sin(a) <= offset
    We find vertices by intersecting adjacent lines in sorted angle order.
    """
    # Sort by angle
    order = np.argsort(angles)
    angles_sorted = angles[order]
    offsets_sorted = offsets[order]
    
    n = len(angles_sorted)
    vertices = []
    
    for i in range(n):
        a1 = angles_sorted[i]
        a2 = angles_sorted[(i+1) % n]
        o1 = offsets_sorted[i]
        o2 = offsets_sorted[(i+1) % n]
        
        # Solve:
        # x*cos(a1) + y*sin(a1) = o1
        # x*cos(a2) + y*sin(a2) = o2
        A = np.array([
            [np.cos(a1), np.sin(a1)],
            [np.cos(a2), np.sin(a2)]
        ])
        b = np.array([o1, o2])
        
        det = A[0,0]*A[1,1] - A[0,1]*A[1,0]
        if abs(det) < 1e-12:
            continue
        
        x = (b[0]*A[1,1] - b[1]*A[0,1]) / det
        y = (A[0,0]*b[1] - A[1,0]*b[0]) / det
        
        # Check if this vertex satisfies all half-plane constraints
        valid = True
        for j in range(n):
            val = x*np.cos(angles_sorted[j]) + y*np.sin(angles_sorted[j])
            if val > offsets_sorted[j] + 1e-9:
                valid = False
                break
        if valid:
            vertices.append([x, y])
    
    if len(vertices) < 3:
        return None
    
    vertices = np.array(vertices)
    return vertices

def _objective(params, n_lines=13):
    """Objective to minimize (negative of score)."""
    angles = params[:n_lines]
    offsets = params[n_lines:2*n_lines]
    
    polygon = _lines_to_polygon(angles, offsets, n_lines)
    if polygon is None:
        return 1e10
    
    # We need exactly 13 points. The polygon has some number of vertices.
    # If it has more than 13, we need to select 13.
    # If it has fewer, we can add interior points or adjust.
    # For simplicity, let's use the polygon vertices directly.
    n_verts = len(polygon)
    
    if n_verts < 3:
        return 1e10
    
    # If we have exactly 13 vertices, great
    # If more, select 13 that maximize min triangle area
    # If fewer, this configuration is invalid for our purposes
    
    # Normalize to unit area
    hull_area = _convex_hull_area(polygon)
    if hull_area <= 1e-10:
        return 1e10
    
    polygon_normalized = polygon / np.sqrt(hull_area)
    
    # Use the first 13 vertices (or all if fewer)
    if n_verts >= 13:
        # Select 13 vertices - for now use evenly spaced
        indices = np.linspace(0, n_verts-1, 13, dtype=int)
        points = polygon_normalized[indices]
    else:
        # Not enough vertices, return bad score
        return 1e10
    
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s

def _local_refine(points, max_iter=50):
    """Local refinement using Nelder-Mead on point positions."""
    n = len(points)
    
    def obj_flat(flat):
        pts = flat.reshape(n, 2)
        s = _score(pts)
        return -s if s > 0 else 1e10
    
    x0 = points.flatten()
    try:
        result = minimize(obj_flat, x0, method='Nelder-Mead', 
                          options={'maxiter': max_iter, 'xatol': 1e-6, 'fatol': 1e-8})
        return result.x.reshape(n, 2)
    except Exception:
        return points

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_score = 0
    best_points = None
    
    # Strategy 1: DE on line parameters
    # 13 angles + 13 offsets = 26 params
    # Angles in [0, 2*pi), offsets in some range
    bounds = [(0, 2*np.pi)] * 13 + [(-5, 5)] * 13
    
    # Run DE with limited time
    de_result = None
    try:
        de_result = differential_evolution(
            _objective, bounds,
            maxiter=20, popsize=15,
            seed=42, tol=1e-4,
            polish=False,
            update_interval=1
        )
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed < deadline - 30 and de_result is not None and de_result.fun < 1e9:
        best_params = de_result.x
        angles = best_params[:13]
        offsets = best_params[13:]
        polygon = _lines_to_polygon(angles, offsets, 13)
        if polygon is not None and len(polygon) >= 13:
            hull_area = _convex_hull_area(polygon)
            if hull_area > 1e-10:
                polygon_norm = polygon / np.sqrt(hull_area)
                n_verts = len(polygon_norm)
                indices = np.linspace(0, n_verts-1, 13, dtype=int)
                points = polygon_norm[indices]
                s = _score(points)
                if s > best_score:
                    best_score = s
                    best_points = points.copy()
        
        # Local refinement
        if best_points is not None and time.time() - start_time < deadline - 10:
            refined = _local_refine(best_points, max_iter=30)
            s_ref = _score(refined)
            if s_ref > best_score:
                best_score = s_ref
                best_points = refined.copy()
    
    # Strategy 2: Start from regular polygon and refine
    if time.time() - start_time < deadline - 15:
        # Regular 13-gon
        theta = np.linspace(0, 2*np.pi, 13, endpoint=False)
        regular_pts = np.column_stack([np.cos(theta), np.sin(theta)])
        s_reg = _score(regular_pts)
        if s_reg > best_score:
            best_score = s_reg
            best_points = regular_pts.copy()
        
        # Refine regular polygon
        if time.time() - start_time < deadline - 10:
            refined = _local_refine(regular_pts, max_iter=50)
            s_ref = _score(refined)
            if s_ref > best_score:
                best_score = s_ref
                best_points = refined.copy()
    
    # Strategy 3: Mixed - some on hull, some interior
    if time.time() - start_time < deadline - 10 and best_points is not None:
        # Try perturbing and re-refining
        rng = np.random.RandomState(42)
        for trial in range(3):
            if time.time() - start_time > deadline - 5:
                break
            perturbed = best_points + rng.normal(0, 0.05, size=best_points.shape)
            refined = _local_refine(perturbed, max_iter=20)
            s = _score(refined)
            if s > best_score:
                best_score = s
                best_points = refined.copy()
    
    if best_points is None:
        # Fallback: regular 13-gon
        theta = np.linspace(0, 2*np.pi, 13, endpoint=False)
        best_points = np.column_stack([np.cos(theta), np.sin(theta)])
    
    return best_points
# EVOLVE-BLOCK-END
