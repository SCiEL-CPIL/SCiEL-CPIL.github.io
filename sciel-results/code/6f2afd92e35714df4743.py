"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START

def _project_to_sphere(x):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms

def _spherical_centroid(points):
    """Compute the spherical centroid of a set of points on the sphere."""
    # Sum the vectors and normalize
    c = np.sum(points, axis=0)
    norm = np.linalg.norm(c)
    if norm < 1e-12:
        # Fallback to uniform random if centroid is near zero (points are antipodal)
        return np.random.randn(3) / np.linalg.norm(np.random.randn(3))
    return c / norm

def _compute_covering_radius(points):
    """
    Compute the covering radius of a set of points on the unit sphere.
    Uses Delaunay triangulation.
    """
    # Ensure points are on the sphere
    points = _project_to_sphere(points)
    
    # Compute Delaunay triangulation
    try:
        tri = Delaunay(points)
    except Exception:
        # Fallback: dense sampling if Delaunay fails (e.g., degenerate)
        M = 1000
        i = np.arange(M) + 0.5
        z = 1.0 - 2.0 * i / M
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        sample_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        dots = np.einsum('ij,kj->ik', sample_points, points)
        max_dots = np.max(dots, axis=1)
        angles = np.arccos(np.clip(max_dots, -1.0, 1.0))
        return np.max(angles)
    
    max_radius = 0.0
    # Iterate over all simplices (triangles)
    for simplex in tri.simplices:
        p1 = points[simplex[0]]
        p2 = points[simplex[1]]
        p3 = points[simplex[2]]
        
        # Compute circumcenter of the spherical triangle
        # The circumcenter is the intersection of the sphere with the plane perpendicular to the vectors?
        # No, the circumcenter of a spherical triangle is the point equidistant from the 3 vertices.
        # It lies on the normal to the plane of the triangle? No.
        # The circumcenter is the intersection of the sphere and the line passing through the origin and the centroid of the triangle's vertices? No.
        
        # A robust way: Solve for c such that |c - p1| = |c - p2| = |c - p3| and |c|=1.
        # This is equivalent to c being in the null space of (p1-p2, p1-p3) and |c|=1?
        # Actually, the circumcenter is the normalized vector:
        # c = (p1 x p2) x (p2 x p3) ... this is getting complicated.
        
        # Simpler: The circumcenter is the intersection of the sphere with the plane defined by the three points? No, that's only if the triangle is right-angled or something.
        
        # Let's use the property that the circumcenter is the center of the unique circle passing through the 3 points on the sphere.
        # This center is the intersection of the sphere with the line perpendicular to the plane of the triangle? No.
        
        # Correct method:
        # The circumcenter is the normalized vector:
        # c = (p1 + p2 + p3) is NOT the circumcenter.
        
        # We can solve the linear system for the circumcenter in 3D space.
        # The circumcenter C satisfies:
        # |C|^2 = 1
        # |C - p1|^2 = |C - p2|^2 = |C - p3|^2
        # Expanding:
        # C^2 - 2 C.p1 + p1^2 = C^2 - 2 C.p2 + p2^2
        # -2 C.p1 + 1 = -2 C.p2 + 1  => C.(p1 - p2) = 0
        # -2 C.p1 + 1 = -2 C.p3 + 1  => C.(p1 - p3) = 0
        
        # So C is orthogonal to (p1-p2) and (p1-p3).
        # C is parallel to (p1-p2) x (p1-p3).
        
        v1 = p1 - p2
        v2 = p1 - p3
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross)
        
        if norm_cross < 1e-10:
            # Degenerate triangle
            continue
            
        c = cross / norm_cross
        # Check if c is on the same side as the points (dot product with centroid)
        # If dot(c, p1) < 0, the triangle is obtuse and the circumcenter is on the "other side"
        # For covering radius, we care about the empty cap.
        # If the circumcenter is inside the triangle, the radius is arccos(dot(c, p1)).
        # If outside, the maximum distance is to the edge.
        
        # Determine if the circumcenter is inside the spherical triangle.
        # A point C is inside the spherical triangle P1P2P3 if the dot products with the normals of the edges (oriented) are consistent.
        # Simpler check: The triangle is acute if the circumcenter is inside.
        # The triangle is acute if all pairwise dot products are less than 1/sqrt(3)? No.
        # The triangle is acute if the circumcenter lies within the triangle.
        
        # Let's check if the circumcenter is "close" to the triangle centroid.
        # Actually, a simpler robust check for the covering radius contribution of a triangle:
        # The covering radius for this cell is the maximum distance from C to the vertices, IF C is inside.
        # If C is outside, the farthest point in the cell is one of the vertices or on the boundary.
        # Actually, for a spherical triangle, the farthest point from the vertices in the Voronoi cell is:
        # - The circumcenter if the triangle is acute.
        # - The midpoint of the longest edge if the triangle is obtuse.
        
        # Check if triangle is acute:
        # Dot products between vertices
        d12 = np.dot(p1, p2)
        d23 = np.dot(p2, p3)
        d13 = np.dot(p1, p3)
        
        # If any dot product is negative, the angle is > 90, so it's obtuse.
        # Wait, angle > 90 means dot < 0.
        # If all dot products are positive, it's acute? Not necessarily, but close.
        # Let's use the circumcenter position.
        # If dot(c, p1) > 0, the circumcenter is in the same hemisphere as p1.
        
        if np.dot(c, p1) > 0:
            # Circumcenter is inside (or at least in the correct hemisphere)
            # The radius is the distance from c to p1
            cos_theta = np.clip(np.dot(c, p1), -1.0, 1.0)
            theta = np.arccos(cos_theta)
            max_radius = max(max_radius, theta)
        else:
            # Circumcenter is outside. The farthest point is likely on the boundary.
            # The maximum distance in this Voronoi cell is the distance to the midpoint of the longest edge.
            # Edges: (p1,p2), (p2,p3), (p1,p3)
            m12 = _spherical_centroid(np.array([p1, p2]))
            m23 = _spherical_centroid(np.array([p2, p3]))
            m13 = _spherical_centroid(np.array([p1, p3]))
            
            # Distance from p1 to m12, etc? No, the distance from the "center" of the cell?
            # The cell is the Voronoi cell. If the circumcenter is outside, the cell is bounded by the perpendicular bisectors of the edges.
            # The farthest point from the vertices in the cell is the intersection of the bisectors that lies farthest from the vertices.
            # This is effectively the midpoint of the longest edge.
            
            # Calculate distances from p1 to the midpoints? No.
            # The covering radius is the max over y of min_i d(y, pi).
            # In the cell of p1, the max of d(y, p1) is the radius.
            # If the circumcenter is outside, the max d(y, p1) in the cell is at the boundary.
            # The boundary is composed of arcs. The farthest point is the midpoint of the longest edge of the triangle.
            
            d12_val = np.arccos(np.clip(d12, -1, 1))
            d23_val = np.arccos(np.clip(d23, -1, 1))
            d13_val = np.arccos(np.clip(d13, -1, 1))
            
            max_edge_len = max(d12_val, d23_val, d13_val)
            # The distance from the vertex to the midpoint of the longest edge is half the edge length?
            # Yes, for the spherical triangle, if it's obtuse, the Voronoi vertex is the midpoint of the longest edge.
            # The distance from the vertex to this midpoint is half the length of the longest edge.
            # Wait, the Voronoi cell of p1 includes the midpoint of the edge (p1, p2) if that edge is part of the hull.
            # The distance from p1 to the midpoint of (p1, p2) is d12/2.
            # Is it possible that the distance to the other vertices is larger?
            # In an obtuse triangle, the circumcenter is outside. The Voronoi cell is bounded by the bisectors of the two shorter edges and the edge itself?
            # Actually, the Voronoi cell of p1 is the set of points closer to p1 than to p2 or p3.
            # The boundary is the bisector of p1-p2 and p1-p3.
            # If the triangle is obtuse at p1, the circumcenter is on the other side of the edge p2-p3.
            # The Voronoi cell is unbounded? No, on a sphere, it's a region.
            # The farthest point in the Voronoi cell of p1 is the midpoint of the longest edge *opposite* to p1? No.
            # It is the midpoint of the longest edge *incident* to p1?
            
            # Let's just compute the max distance to the midpoints of the edges incident to p1.
            # And also check the distance to the other vertices? No, they are further.
            
            # The candidate radius for this cell is max(d12/2, d13/2).
            # But we must ensure this point is actually in the cell.
            # For a convex hull, this should hold.
            
            cell_radius = max(d12_val / 2.0, d13_val / 2.0)
            max_radius = max(max_radius, cell_radius)
            
    return max_radius

def _lloyd_step(points):
    """Perform one step of Lloyd's algorithm."""
    points = _project_to_sphere(points)
    
    try:
        tri = Delaunay(points)
    except Exception:
        return points
    
    new_points = points.copy()
    
    # For each point, find the triangles it belongs to
    # tri.simplices has shape (N_tri, 3)
    # We need to map point index to list of triangle indices
    
    point_to_tris = {i: [] for i in range(N)}
    for idx, simplex in enumerate(tri.simplices):
        for p_idx in simplex:
            point_to_tris[p_idx].append(idx)
    
    for i in range(N):
        tris = point_to_tris[i]
        if not tris:
            continue
            
        # Collect all vertices of the adjacent triangles
        # The Voronoi cell of point i is the union of the spherical triangles formed by i and its neighbors?
        # No, the Voronoi cell is a spherical polygon.
        # Approximation: Move to the centroid of the neighboring points.
        # Better: Move to the centroid of the vertices of the adjacent triangles.
        
        neighbors = set()
        for t_idx in tris:
            simplex = tri.simplices[t_idx]
            for p_idx in simplex:
                if p_idx != i:
                    neighbors.add(p_idx)
        
        if not neighbors:
            continue
            
        neighbor_points = points[list(neighbors)]
        centroid = _spherical_centroid(neighbor_points)
        
        # Move towards centroid
        # Blend to avoid oscillation
        alpha = 0.5
        new_points[i] = (1 - alpha) * points[i] + alpha * centroid
    
    return _project_to_sphere(new_points)

def _random_points(n):
    """Generate n random points on the sphere."""
    pts = np.random.randn(n, 3)
    return _project_to_sphere(pts)

def _fibonacci_spiral(n):
    """Generate n points on the sphere using a Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _project_to_sphere(x)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # Initial configuration: Fibonacci spiral
    x = _fibonacci_spiral(N)
    best_x = x.copy()
    best_radius = _compute_covering_radius(x)
    
    # Lloyd's Algorithm
    max_lloyd_iters = 200
    for it in range(max_lloyd_iters):
        if time.time() > deadline - 30.0:
            break
            
        x_new = _lloyd_step(x)
        
        # Check for improvement
        radius_new = _compute_covering_radius(x_new)
        
        if radius_new < best_radius:
            best_radius = radius_new
            best_x = x_new.copy()
            x = x_new
        else:
            # If no improvement, try a perturbation
            # This helps escape local minima
            if it % 20 == 19:
                x_pert = x + 0.1 * np.random.randn(N, 3)
                x_pert = _project_to_sphere(x_pert)
                radius_pert = _compute_covering_radius(x_pert)
                if radius_pert < best_radius:
                    best_radius = radius_pert
                    best_x = x_pert.copy()
                    x = x_pert
    
    # Local Optimization using Gradient Descent on the covering radius
    # We use a smooth approximation of the covering radius for gradients
    # Or we can use scipy.optimize.minimize on a discrete set of sample points
    
    # Let's do a simple local search
    x = best_x.copy()
    lr = 0.05
    for it in range(500):
        if time.time() > deadline - 5.0:
            break
        
        # Compute gradient of the max distance to a dense set of sample points
        M = 1000
        i = np.arange(M) + 0.5
        z = 1.0 - 2.0 * i / M
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        sample_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
        # For each sample point, find the nearest config point
        dots = np.einsum('ij,kj->ik', sample_points, x)
        max_dots = np.max(dots, axis=1)
        nearest_idx = np.argmax(dots, axis=1)
        
        # Gradient of arccos(dot) wrt x
        # d(arccos(d))/dx = -1/sqrt(1-d^2) * dy
        # Only the nearest point contributes to the max
        
        grad = np.zeros_like(x)
        for i_sample in range(M):
            idx = nearest_idx[i_sample]
            d = dots[i_sample, idx]
            d = np.clip(d, -0.999999, 0.999999)
            factor = -1.0 / np.sqrt(1.0 - d**2)
            grad[idx] += factor * sample_points[i_sample]
            
        # Normalize gradient
        grad_norm = np.linalg.norm(grad, axis=1, keepdims=True)
        grad_norm = np.maximum(grad_norm, 1e-12)
        grad = grad / grad_norm
        
        # Backtracking line search
        step = lr
        improved = False
        for _ in range(10):
            x_trial = x - step * grad
            x_trial = _project_to_sphere(x_trial)
            
            # Check for coincident points
            min_dist = np.min(np.linalg.norm(x_trial[:, None, :] - x_trial[None, :, :], axis=2))
            if min_dist < 1e-6:
                step *= 0.5
                continue
                
            radius_trial = _compute_covering_radius(x_trial)
            if radius_trial < best_radius:
                x = x_trial
                best_radius = radius_trial
                best_x = x.copy()
                improved = True
                break
            else:
                step *= 0.5
                
        if not improved:
            # Try a small random perturbation
            x_pert = x + 0.01 * np.random.randn(N, 3)
            x_pert = _project_to_sphere(x_pert)
            radius_pert = _compute_covering_radius(x_pert)
            if radius_pert < best_radius:
                best_radius = radius_pert
                best_x = x_pert.copy()
                x = x_pert
    
    return _project_to_sphere(best_x)

# EVOLVE-BLOCK-END
