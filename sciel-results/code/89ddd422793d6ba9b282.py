import time
import numpy as np
from scipy.spatial import ConvexHull, Delaunay

N = 130
_EPS = 1e-12

def _initial_fib():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_init(rng):
    """Generate random initial points."""
    pts = rng.standard_normal((N, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _spherical_centroid(vertices):
    """
    Compute the spherical centroid of a set of vertices on the unit sphere.
    This is the normalized sum of the vertices, which approximates the centroid
    of the spherical polygon for small cells. For larger cells, a more accurate
    method would be needed, but this is a good first-order approximation.
    """
    if len(vertices) == 0:
        return None
    vec = np.sum(vertices, axis=0)
    norm = np.linalg.norm(vec)
    if norm < _EPS:
        return None
    return vec / norm

def _compute_hull_and_radius(pts):
    """
    Compute the convex hull, the covering radius, and the facet data.
    Returns: (radius, hull, faces, normals, poles, facet_vertices)
    """
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[:, 2]]
        
        # Compute normals for each face
        AB = B - A
        AC = C - A
        cross = np.cross(AB, AC)
        norms = np.linalg.norm(cross, axis=1, keepdims=True)
        
        # Filter degenerate
        valid = norms.flatten() > _EPS
        if not np.any(valid):
            return np.pi, None, None, None, None, None
            
        normals = cross[valid] / (norms[valid] + _EPS)
        
        # Ensure outward direction
        centroids = (A[valid] + B[valid] + C[valid]) / 3.0
        signs = np.sum(normals * centroids, axis=1, keepdims=True)
        normals = normals * np.sign(signs)
        
        # The pole is the normalized normal.
        # Distance is arccos(pole . vertex)
        cos_theta = np.sum(normals * A[valid], axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        radii = np.arccos(cos_theta)
        max_r = np.max(radii)
        
        # Identify the worst facet
        worst_idx = np.argmax(radii)
        
        # Poles are the normals themselves (normalized)
        poles = normals
        
        # Facet vertices for the valid facets
        valid_faces = faces[valid]
        
        return max_r, hull, valid_faces, normals, poles, valid_faces
        
    except Exception:
        return np.pi, None, None, None, None, None

def _spherical_lloyd_step(pts, step_size=1.0):
    """
    Perform one step of Spherical Lloyd's Algorithm.
    1. Compute Voronoi cells using Delaunay triangulation.
    2. Move each point to the centroid of its cell.
    """
    try:
        # Compute Delaunay triangulation on the sphere
        # scipy.spatial.Delaunay works in 3D, but we need the spherical version.
        # A common trick is to project to 2D or use the convex hull.
        # However, a more direct method for spherical Voronoi is to use the dual of the convex hull.
        # The convex hull faces are the Delaunay triangles, and the vertices of the hull are the points.
        # The Voronoi cell of a point is the set of points on the sphere closer to it than to any other.
        # The vertices of the Voronoi cell are the poles of the Delaunay triangles incident to the point.
        
        hull = ConvexHull(pts)
        faces = hull.simplices
        
        # For each point, find the faces incident to it
        # Then, the vertices of its Voronoi cell are the normals of those faces.
        # Compute normals for all faces
        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[:, 2]]
        
        AB = B - A
        AC = C - A
        cross = np.cross(AB, AC)
        norms = np.linalg.norm(cross, axis=1, keepdims=True)
        
        valid = norms.flatten() > _EPS
        if not np.all(valid):
            return pts
            
        normals = cross[valid] / (norms[valid] + _EPS)
        
        # Ensure outward direction
        centroids = (A[valid] + B[valid] + C[valid]) / 3.0
        signs = np.sum(normals * centroids, axis=1, keepdims=True)
        normals = normals * np.sign(signs)
        
        valid_faces = faces[valid]
        
        # Map from point index to list of face indices
        point_to_faces = [[] for _ in range(N)]
        for i, face in enumerate(valid_faces):
            for v in face:
                point_to_faces[v].append(i)
        
        new_pts = pts.copy()
        for i in range(N):
            if len(point_to_faces[i]) == 0:
                continue
                
            # Get the normals of the incident faces (these are the Voronoi cell vertices)
            face_indices = point_to_faces[i]
            voronoi_vertices = normals[face_indices]
            
            # Compute the spherical centroid of these vertices
            centroid = _spherical_centroid(voronoi_vertices)
            if centroid is None:
                continue
                
            # Move the point towards the centroid
            # Use a step size to avoid overshooting
            new_pts[i] = (1.0 - step_size) * pts[i] + step_size * centroid
            new_pts[i] /= (np.linalg.norm(new_pts[i]) + _EPS)
            
        return new_pts
        
    except Exception:
        return pts

def _pole_chase(pts, step_size=0.1):
    """
    Identify the worst-covered region (the pole of the largest facet) and
    move the three vertices of that facet toward the pole's circumcenter.
    This is a targeted move to escape local minima.
    """
    radius, hull, faces, normals, poles, valid_faces = _compute_hull_and_radius(pts)
    
    if hull is None:
        return pts
        
    # Find the worst facet
    A = pts[faces[:, 0]]
    cos_theta = np.sum(normals * A, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    radii = np.arccos(cos_theta)
    worst_idx = np.argmax(radii)
    
    worst_face = valid_faces[worst_idx]
    pole = normals[worst_idx]
    
    # The circumcenter of the spherical triangle is the pole itself.
    # Move the three vertices toward the pole.
    new_pts = pts.copy()
    for v in worst_face:
        new_pts[v] = (1.0 - step_size) * pts[v] + step_size * pole
        new_pts[v] /= (np.linalg.norm(new_pts[v]) + _EPS)
        
    return new_pts

def _local_polish(pts, n_iters=50):
    """
    Alternate between Spherical Lloyd's steps and Pole Chasing.
    """
    for i in range(n_iters):
        # Lloyd step
        pts = _spherical_lloyd_step(pts, step_size=0.5)
        
        # Pole chase step (less frequent)
        if i % 5 == 0:
            pts = _pole_chase(pts, step_size=0.05)
            
    return pts

def construct():
    deadline = time.time() + 145.0
    rng = np.random.default_rng(42)
    
    best_pts = _initial_fib()
    best_r, _, _, _, _, _ = _compute_hull_and_radius(best_pts)
    
    n_restarts = 15
    
    for restart in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
            
        if restart == 0:
            pts = _initial_fib()
        else:
            pts = _random_init(rng)
            
        # Local polish
        phase_deadline = time.time() + 8.0
        iters = 0
        while time.time() < phase_deadline and iters < 100:
            pts = _local_polish(pts, n_iters=10)
            iters += 10
            
        # Evaluate
        curr_r, _, _, _, _, _ = _compute_hull_and_radius(pts)
        if curr_r < best_r:
            best_r = curr_r
            best_pts = pts.copy()
            
    # Final polish on the best configuration
    if time.time() < deadline - 5.0:
        phase_deadline = time.time() + 10.0
        while time.time() < phase_deadline:
            best_pts = _local_polish(best_pts, n_iters=10)
            curr_r, _, _, _, _, _ = _compute_hull_and_radius(best_pts)
            if curr_r < best_r:
                best_r = curr_r
            else:
                # If no improvement, break to save time
                break
                
    # Final normalization
    best_pts = np.array(best_pts, dtype=float)
    best_pts /= (np.linalg.norm(best_pts, axis=1, keepdims=True) + _EPS)
    
    if not np.all(np.isfinite(best_pts)) or best_pts.shape != (N, 3):
        return _initial_fib()
        
    return best_pts
