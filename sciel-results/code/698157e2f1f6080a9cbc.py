import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 130

def _compute_covering_radius(points):
    """Compute exact covering radius via convex hull facet circumradii."""
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi
    
    faces = hull.simplices
    max_circ = 0.0
    
    for face in faces:
        a, b, c = points[face[0]], points[face[1]], points[face[2]]
        # Compute circumcenter of triangle on unit sphere
        # For a spherical triangle, the circumcenter is the normalized cross product approach
        # or we can compute the Voronoi vertex (pole of the great circle through the face)
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            continue
        d = np.dot(n, a)
        t = d / (n_norm ** 2)
        o = t * n
        o_norm = np.linalg.norm(o)
        if o_norm < 1e-10:
            continue
        o = o / o_norm
        
        # Check if this Voronoi vertex is outside the triangle (invalid)
        # For a valid facet, the Voronoi vertex should be on the same side as the triangle
        # Compute angles from each vertex to the Voronoi vertex
        dots = np.array([np.dot(a, o), np.dot(b, o), np.dot(c, o)])
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(dots)
        circumradius = np.max(angles)
        
        if circumradius > max_circ:
            max_circ = circumradius
    
    return max_circ

def _compute_facet_circumradii_and_centers(points):
    """
    For each point, compute the maximum circumradius among its adjacent facets,
    and the direction (toward the Voronoi vertex of the worst facet) to move.
    Returns:
        max_circ_per_point: (N,) array of max circumradius per point
        move_dirs: (N, 3) array of unit vectors indicating where to move
    """
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.ones(N) * np.pi, np.zeros((N, 3))
    
    faces = hull.simplices
    max_circ_per_point = np.zeros(N)
    move_dirs = np.zeros((N, 3))
    
    for face in faces:
        a, b, c = points[face[0]], points[face[1]], points[face[2]]
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            continue
        d = np.dot(n, a)
        t = d / (n_norm ** 2)
        o = t * n
        o_norm = np.linalg.norm(o)
        if o_norm < 1e-10:
            continue
        o = o / o_norm
        
        dots = np.array([np.dot(a, o), np.dot(b, o), np.dot(c, o)])
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(dots)
        circumradius = np.max(angles)
        
        for idx, angle in zip(face, angles):
            if angle >= max_circ_per_point[idx]:
                max_circ_per_point[idx] = angle
                # Move toward the Voronoi vertex of this facet
                move_dirs[idx] = o
    
    # For points with no adjacent facets (shouldn't happen for convex hull), keep zero
    norms = np.linalg.norm(move_dirs, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    move_dirs = move_dirs / norms
    
    return max_circ_per_point, move_dirs

def _facet_based_optimize(points, max_iters=200, deadline=None):
    """
    Lloyd-type optimization: move each point toward the Voronoi vertex of its worst facet.
    Uses adaptive step size based on current covering radius.
    """
    points = points.copy()
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        max_circ, move_dirs = _compute_facet_circumradii_and_centers(points)
        current_radius = np.max(max_circ)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
        
        # Adaptive step size: smaller steps when close to optimum
        step = 0.1 * np.sqrt(current_radius)
        step = np.clip(step, 0.001, 0.1)
        
        # Move points toward their worst facet Voronoi vertices
        new_points = points + step * move_dirs
        new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
        
        # Check if this improved things; if not, try smaller step
        new_radius = _compute_covering_radius(new_points)
        if new_radius < current_radius:
            points = new_points
        else:
            # Try half step
            new_points = points + 0.5 * step * move_dirs
            new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
            new_radius = _compute_covering_radius(new_points)
            if new_radius < current_radius:
                points = new_points
            else:
                # Try quarter step
                new_points = points + 0.25 * step * move_dirs
                new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
                new_radius = _compute_covering_radius(new_points)
                if new_radius < current_radius:
                    points = new_points
                else:
                    break  # No improvement possible with this step
    
    return best_points, best_radius

def _nelder_mead_polish(points, max_iters=50, deadline=None):
    """
    Polish the configuration using Nelder-Mead on the exact covering radius.
    Focuses on reducing the maximum facet circumradius.
    """
    points = points.copy()
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    def objective(x):
        if deadline is not None and time.time() > deadline:
            return best_radius
        pts = x.reshape(N, 3)
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        return _compute_covering_radius(pts)
    
    try:
        res = minimize(
            objective,
            points.flatten(),
            method='Nelder-Mead',
            options={'maxiter': max_iters, 'xatol': 1e-10, 'fatol': 1e-10}
        )
        
        if res.fun < best_radius:
            new_points = res.x.reshape(N, 3)
            new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
            new_radius = _compute_covering_radius(new_points)
            if new_radius < best_radius:
                best_points = new_points
                best_radius = new_radius
    except Exception:
        pass
    
    return best_points, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    # 1. Initial Guess: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    initial_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = initial_points.copy()
    best_radius = _compute_covering_radius(best_points)
    
    # 2. Phase 1: Facet-based Lloyd optimization
    if time.time() < deadline - 30.0:
        points, radius = _facet_based_optimize(initial_points, max_iters=100, deadline=deadline - 30.0)
        if radius < best_radius:
            best_points = points
            best_radius = radius
    
    # 3. Phase 2: Nelder-Mead polish on the best configuration
    if time.time() < deadline - 20.0:
        points, radius = _nelder_mead_polish(best_points, max_iters=50, deadline=deadline - 10.0)
        if radius < best_radius:
            best_points = points
            best_radius = radius
    
    # 4. Phase 3: Another round of facet-based optimization from the polished result
    if time.time() < deadline - 15.0:
        points, radius = _facet_based_optimize(best_points, max_iters=50, deadline=deadline - 10.0)
        if radius < best_radius:
            best_points = points
            best_radius = radius
    
    # 5. Final Nelder-Mead polish
    if time.time() < deadline - 5.0:
        points, radius = _nelder_mead_polish(best_points, max_iters=30, deadline=deadline - 2.0)
        if radius < best_radius:
            best_points = points
            best_radius = radius
    
    # Ensure output is normalized
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
