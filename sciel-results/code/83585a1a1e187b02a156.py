import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    return pts / norms

def _circumcenter_robust(p1, p2, p3):
    """Compute circumcenter of triangle on unit sphere, robust to degeneracy."""
    # Use vector algebra: circumcenter is perpendicular to the plane through p1,p2,p3
    # and equidistant from all three points
    # For unit sphere, if triangle is acute, circumcenter = normalized normal
    # If obtuse, circumcenter is on opposite side
    
    # Compute normal to the triangle
    v1 = p2 - p1
    v2 = p3 - p1
    n = np.cross(v1, v2)
    n_norm = np.linalg.norm(n)
    
    if n_norm < 1e-12:
        # Degenerate triangle - return centroid
        center = (p1 + p2 + p3) / 3.0
        norm = np.linalg.norm(center)
        if norm < 1e-12:
            return p1.copy()
        return center / norm
    
    n = n / n_norm
    
    # Check if circumcenter is on the same side as the triangle
    # by checking if dot products with all vertices are positive
    dot1 = np.dot(n, p1)
    dot2 = np.dot(n, p2)
    dot3 = np.dot(n, p3)
    
    if dot1 > 0 and dot2 > 0 and dot3 > 0:
        return n
    else:
        return -n

def _covering_radius_deg(points):
    """Compute the covering radius in degrees using convex hull."""
    try:
        hull = ConvexHull(points)
    except Exception:
        return 180.0  # Invalid hull, return large value
        
    max_radius = 0.0
    simplices = hull.simplices
    
    for simplex in simplices:
        p1 = points[simplex[0]]
        p2 = points[simplex[1]]
        p3 = points[simplex[2]]
        
        # Compute circumcenter
        center = _circumcenter_robust(p1, p2, p3)
        
        # The covering radius for this facet is the angular distance from center to any vertex
        dist1 = np.arccos(np.clip(np.dot(center, p1), -1.0, 1.0))
        max_radius = max(max_radius, dist1)
    
    return np.degrees(max_radius)

def _fibonacci_spiral(n):
    """Generate n points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)

def _compute_voronoi_centroids(points):
    """Compute centroid of each Voronoi cell on the sphere.
    
    For each point, find its Voronoi cell (points on sphere closer to it than to any other),
    then compute the centroid of that cell.
    """
    n = len(points)
    centroids = np.zeros_like(points)
    
    # Create a fine grid of points on the sphere
    # Use icosahedron subdivision for better coverage
    def _icosahedron_subdivide(subdivisions=5):
        # Golden ratio
        t = (1 + np.sqrt(5)) / 2
        # Icosahedron vertices
        verts = np.array([
            [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
            [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
            [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
        ], dtype=float)
        verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
        
        # Icosahedron faces
        faces = [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
        ]
        
        # Subdivide
        for _ in range(subdivisions):
            new_faces = []
            for face in faces:
                a, b, c = verts[face[0]], verts[face[1]], verts[face[2]]
                # Midpoints
                ab = (a + b) / np.linalg.norm(a + b)
                bc = (b + c) / np.linalg.norm(b + c)
                ca = (c + a) / np.linalg.norm(c + a)
                # Add new vertices
                verts = np.vstack([verts, [ab, bc, ca]])
                # New faces
                new_faces.extend([[face[0], len(verts)-3, len(verts)-1],
                                  [face[1], len(verts)-1, len(verts)-2],
                                  [face[2], len(verts)-2, len(verts)-3],
                                  [len(verts)-3, len(verts)-2, len(verts)-1]])
            faces = new_faces
        
        return verts
    
    # Use a reasonable number of sample points
    sample_pts = _icosahedron_subdivide(6)  # ~2562 points
    
    # For each configuration point, find all sample points closer to it than to any other
    for i in range(n):
        # Compute distances from all sample points to point i and all other points
        dist_i = np.arccos(np.clip(sample_pts @ points[i], -1.0, 1.0))
        
        # Find minimum distance to any other point
        min_dist_other = np.full(len(sample_pts), np.inf)
        for j in range(n):
            if j == i:
                continue
            dist_j = np.arccos(np.clip(sample_pts @ points[j], -1.0, 1.0))
            min_dist_other = np.minimum(min_dist_other, dist_j)
        
        # Sample points in Voronoi cell of point i
        mask = dist_i < min_dist_other
        if np.any(mask):
            centroids[i] = np.sum(sample_pts[mask], axis=0)
        
        # Normalize
        norm = np.linalg.norm(centroids[i])
        if norm > 1e-12:
            centroids[i] = centroids[i] / norm
        else:
            centroids[i] = points[i]
    
    return centroids

def _optimize(pts, deadline):
    """Optimize using hybrid approach: facet-based + Voronoi relaxation."""
    current_pts = pts.copy()
    best_pts = pts.copy()
    best_radius = _covering_radius_deg(pts)
    
    # Phase 1: Facet-based optimization (fast, local)
    step_size = 0.1
    no_improve_count = 0
    max_no_improve = 15
    
    # Use Voronoi relaxation periodically
    use_voronoi = True
    voronoi_interval = 20  # Every 20 iterations, try Voronoi
    
    iteration = 0
    while time.time() < deadline - 30.0:  # Leave time for final polish
        # Decide whether to use Voronoi or facet approach
        if use_voronoi and iteration % voronoi_interval == 0 and time.time() < deadline - 60.0:
            # Voronoi relaxation
            try:
                centroids = _compute_voronoi_centroids(current_pts)
                new_pts = current_pts.copy()
                for i in range(N):
                    direction = centroids[i] - current_pts[i]
                    dir_norm = np.linalg.norm(direction)
                    if dir_norm > 1e-10:
                        direction = direction / dir_norm
                        new_pts[i] = current_pts[i] + 0.3 * direction
                        new_pts[i] = new_pts[i] / np.linalg.norm(new_pts[i])
                
                new_radius = _covering_radius_deg(new_pts)
                if new_radius < best_radius - 1e-8:
                    best_radius = new_radius
                    best_pts = new_pts.copy()
                    current_pts = new_pts.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            except Exception:
                no_improve_count += 1
        else:
            # Facet-based: find largest facet and move vertices toward circumcenter
            try:
                hull = ConvexHull(current_pts)
            except Exception:
                break
                
            max_radius = 0.0
            max_simplex = None
            max_center = None
            
            for simplex in hull.simplices:
                p1 = current_pts[simplex[0]]
                p2 = current_pts[simplex[1]]
                p3 = current_pts[simplex[2]]
                
                center = _circumcenter_robust(p1, p2, p3)
                dist = np.arccos(np.clip(np.dot(center, p1), -1.0, 1.0))
                
                if dist > max_radius:
                    max_radius = dist
                    max_simplex = simplex
                    max_center = center
            
            if max_simplex is None:
                break
            
            # Move the three vertices toward the circumcenter
            new_pts = current_pts.copy()
            for idx in max_simplex:
                p = current_pts[idx]
                direction = max_center - p
                dir_norm = np.linalg.norm(direction)
                if dir_norm < 1e-10:
                    continue
                direction = direction / dir_norm
                new_pts[idx] = p + step_size * direction
                new_pts[idx] = new_pts[idx] / np.linalg.norm(new_pts[idx])
            
            # Evaluate new configuration
            new_radius = _covering_radius_deg(new_pts)
            
            if new_radius < best_radius - 1e-8:
                best_radius = new_radius
                best_pts = new_pts.copy()
                current_pts = new_pts.copy()
                no_improve_count = 0
                step_size = min(step_size * 1.03, 0.2)
            else:
                no_improve_count += 1
                step_size = max(step_size * 0.95, 0.02)
                
                if no_improve_count >= max_no_improve:
                    # Try perturbation to escape local minimum
                    if time.time() < deadline - 40.0:
                        current_pts = best_pts.copy()
                        n_perturb = max(5, N // 8)
                        idxs = np.random.choice(N, n_perturb, replace=False)
                        for idx in idxs:
                            p = current_pts[idx]
                            rand_vec = np.random.randn(3)
                            tangent = rand_vec - np.dot(rand_vec, p) * p
                            tangent_norm = np.linalg.norm(tangent)
                            if tangent_norm < 1e-10:
                                continue
                            tangent = tangent / tangent_norm
                            current_pts[idx] = p + 0.08 * tangent
                            current_pts[idx] = current_pts[idx] / np.linalg.norm(current_pts[idx])
                        
                        no_improve_count = 0
                        step_size = 0.1
                    else:
                        break
        
        iteration += 1
    
    # Phase 2: Final polish with small steps
    current_pts = best_pts.copy()
    best_radius = _covering_radius_deg(current_pts)
    step_size = 0.01
    polish_iterations = 0
    max_polish = 100
    
    while time.time() < deadline - 5.0 and polish_iterations < max_polish:
        try:
            hull = ConvexHull(current_pts)
        except Exception:
            break
            
        max_radius = 0.0
        max_simplex = None
        max_center = None
        
        for simplex in hull.simplices:
            p1 = current_pts[simplex[0]]
            p2 = current_pts[simplex[1]]
            p3 = current_pts[simplex[2]]
            
            center = _circumcenter_robust(p1, p2, p3)
            dist = np.arccos(np.clip(np.dot(center, p1), -1.0, 1.0))
            
            if dist > max_radius:
                max_radius = dist
                max_simplex = simplex
                max_center = center
        
        if max_simplex is None:
            break
        
        new_pts = current_pts.copy()
        for idx in max_simplex:
            p = current_pts[idx]
            direction = max_center - p
            dir_norm = np.linalg.norm(direction)
            if dir_norm < 1e-10:
                continue
            direction = direction / dir_norm
            new_pts[idx] = p + step_size * direction
            new_pts[idx] = new_pts[idx] / np.linalg.norm(new_pts[idx])
        
        new_radius = _covering_radius_deg(new_pts)
        
        if new_radius < best_radius - 1e-9:
            best_radius = new_radius
            best_pts = new_pts.copy()
            current_pts = new_pts.copy()
        else:
            step_size *= 0.8
            if step_size < 0.001:
                break
        
        polish_iterations += 1
    
    return best_pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Try multiple initializations and keep the best
    best_pts = None
    best_radius = np.inf
    
    # Initialization 1: Fibonacci spiral
    pts1 = _fibonacci_spiral(N)
    try:
        optimized1 = _optimize(pts1, deadline - 30.0)  # Leave time for other runs
        radius1 = _covering_radius_deg(optimized1)
        if radius1 < best_radius:
            best_radius = radius1
            best_pts = optimized1.copy()
    except Exception:
        pass
    
    # Initialization 2: Random points (if time allows)
    if time.time() < deadline - 40.0:
        np.random.seed(42)
        pts2 = np.random.randn(N, 3)
        pts2 = _normalize(pts2)
        try:
            optimized2 = _optimize(pts2, deadline - 20.0)
            radius2 = _covering_radius_deg(optimized2)
            if radius2 < best_radius:
                best_radius = radius2
                best_pts = optimized2.copy()
        except Exception:
            pass
    
    # Return best result
    if best_pts is not None and np.all(np.isfinite(best_pts)):
        return _normalize(best_pts)
    
    # Fallback
    return _normalize(pts1)
