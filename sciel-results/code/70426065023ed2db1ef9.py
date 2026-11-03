import numpy as np
from scipy.spatial import ConvexHull
import time
from scipy.optimize import minimize

N = 130

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    return pts / norms

def _fibonacci_spiral(n, seed=42):
    """Generate n points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)

def _spherical_circumcenter(tri):
    """
    Compute the spherical circumcenter of a triangle on the unit sphere.
    tri: (3, 3) array of unit vectors.
    Returns unit vector circumcenter.
    """
    a, b, c = tri
    # Vector v = (a-b) x (b-c) is parallel to the spherical circumcenter
    v = np.cross(a - b, b - c)
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        # Degenerate: points are collinear or coincident
        # Use centroid
        center = (a + b + c) / 3.0
        norm = np.linalg.norm(center)
        if norm < 1e-10:
            return a # Fallback
        return center / norm
    return v / norm

def _covering_radius(pts):
    """
    Compute the covering radius of a set of points on the unit sphere.
    Uses convex hull facets.
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi / 2.0 # Fallback: worst case for random
    
    max_radius = 0.0
    for simplex in hull.simplices:
        tri = pts[simplex]
        center = _spherical_circumcenter(tri)
        
        # The circumradius is the angular distance from center to any vertex
        # We use the first vertex
        dot = np.clip(np.dot(center, tri[0]), -1.0, 1.0)
        radius = np.arccos(dot)
        
        if radius > max_radius:
            max_radius = radius
            
    return max_radius

def _lloyd_iteration(pts):
    """
    Perform one Lloyd's algorithm iteration.
    Moves each point towards the centroid of its Voronoi cell.
    For spherical Voronoi, the 'centroid' is approximated by the circumcenter
    of the worst facet of the cell, or the average of the circumcenters.
    Here we use the circumcenter of the facet with the largest circumradius
    within the point's cell, to specifically target covering radius reduction.
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return pts.copy()
    
    new_pts = pts.copy()
    
    # Map point index to list of simplex indices
    point_to_simplices = {i: [] for i in range(len(pts))}
    for idx, simplex in enumerate(hull.simplices):
        for v in simplex:
            point_to_simplices[v].append(idx)
            
    for i in range(len(pts)):
        simplices = point_to_simplices[i]
        if not simplices:
            continue
            
        # Find the worst facet for this point
        worst_center = None
        worst_radius = -1.0
        for s_idx in simplices:
            tri = pts[hull.simplices[s_idx]]
            center = _spherical_circumcenter(tri)
            dot = np.clip(np.dot(center, tri[0]), -1.0, 1.0)
            radius = np.arccos(dot)
            if radius > worst_radius:
                worst_radius = radius
                worst_center = center
                
        if worst_center is not None:
            # Move towards the center
            # Step size control: move a fraction of the way
            # If the point is far, move less?
            # Simple: move 0.5 of the way in tangent space
            diff = worst_center - pts[i]
            # Project to tangent space
            tangent = diff - np.dot(diff, pts[i]) * pts[i]
            new_pts[i] = pts[i] + 0.5 * tangent
            new_pts[i] = new_pts[i] / np.linalg.norm(new_pts[i])
            
    return new_pts

def _nelder_mead_polish(pts, deadline):
    """
    Polish the configuration by applying Nelder-Mead to the vertices
    of the worst facet.
    """
    current_pts = pts.copy()
    best_radius = _covering_radius(current_pts)
    
    while time.time() < deadline:
        try:
            hull = ConvexHull(current_pts)
        except Exception:
            break
            
        # Find worst facet
        max_radius = 0.0
        worst_facet = None
        for simplex in hull.simplices:
            tri = current_pts[simplex]
            center = _spherical_circumcenter(tri)
            dot = np.clip(np.dot(center, tri[0]), -1.0, 1.0)
            radius = np.arccos(dot)
            if radius > max_radius:
                max_radius = radius
                worst_facet = simplex
                
        if worst_facet is None:
            break
            
        # Extract indices
        i, j, k = worst_facet
        
        # Define objective function for Nelder-Mead
        # We move points i, j, k on the sphere.
        # Parameters: angles/offsets? 
        # Simpler: use a local coordinate system or just perturb and project.
        # scipy minimize works on R^n. We can map 3D points to 2D tangent coordinates?
        # Or just use a simple custom Nelder-Mead that projects back to sphere.
        
        def objective(params):
            # params: 6 floats (dx, dy, dz for i, j, k)
            p_i = current_pts[i] + params[0:3]
            p_j = current_pts[j] + params[3:6]
            p_k = current_pts[k] + params[6:9]
            
            # Normalize
            p_i = p_i / np.linalg.norm(p_i)
            p_j = p_j / np.linalg.norm(p_j)
            p_k = p_k / np.linalg.norm(p_k)
            
            # Update points in a temporary array
            temp_pts = current_pts.copy()
            temp_pts[i] = p_i
            temp_pts[j] = p_j
            temp_pts[k] = p_k
            
            # Check validity: no coincident points?
            # Covering radius
            return _covering_radius(temp_pts)
            
        # Initial params: 0
        x0 = np.zeros(9)
        
        # Run Nelder-Mead
        try:
            res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-6})
        except Exception:
            break
            
        if res.fun < best_radius:
            # Apply update
            current_pts[i] = (current_pts[i] + res.x[0:3])
            current_pts[j] = (current_pts[j] + res.x[3:6])
            current_pts[k] = (current_pts[k] + res.x[6:9])
            current_pts[i] = current_pts[i] / np.linalg.norm(current_pts[i])
            current_pts[j] = current_pts[j] / np.linalg.norm(current_pts[j])
            current_pts[k] = current_pts[k] / np.linalg.norm(current_pts[k])
            best_radius = res.fun
        else:
            # If no improvement, break or try another facet?
            # For now, break to avoid cycling
            break
            
    return current_pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # 1. Initialize
    pts = _fibonacci_spiral(N)
    best_pts = pts.copy()
    best_radius = _covering_radius(best_pts)
    
    # 2. Lloyd's Algorithm Phase
    # Run for a fixed number of iterations or time limit
    lloyd_deadline = start_time + 60.0
    while time.time() < lloyd_deadline:
        new_pts = _lloyd_iteration(best_pts)
        new_radius = _covering_radius(new_pts)
        if new_radius < best_radius:
            best_radius = new_radius
            best_pts = new_pts
        else:
            # If no improvement, break Lloyd's phase
            break
            
    # 3. Nelder-Mead Polishing Phase
    nm_deadline = deadline - 5.0
    polished_pts = _nelder_mead_polish(best_pts, nm_deadline)
    polished_radius = _covering_radius(polished_pts)
    
    if polished_radius < best_radius:
        best_pts = polished_pts
        best_radius = polished_radius
        
    return _normalize(best_pts)
