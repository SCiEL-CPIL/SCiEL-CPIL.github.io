import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    best_points = points.copy()
    best_radius = np.inf

    # Function to compute covering radius
    def compute_covering_radius(pts):
        try:
            hull = ConvexHull(pts)
            facets = hull.simplices
            max_rad = 0.0
            for tri in facets:
                p1, p2, p3 = pts[tri]
                # Compute circumcenter of the triangle on the sphere
                # The circumcenter is the intersection of the perpendicular bisectors
                # In 3D, it's the point equidistant from p1, p2, p3 on the sphere
                # We can solve for the circumcenter by finding the point c such that
                # |c - p1| = |c - p2| = |c - p3| and |c| = 1
                # This is equivalent to c being in the plane of p1, p2, p3 and equidistant
                # The circumcenter of a spherical triangle is the normalized cross product
                # of the normals of the great circles, but a simpler way is to use the
                # fact that the circumcenter is the pole of the great circle through the
                # three points, but that's for the circle, not the triangle.
                
                # Actually, the circumcenter of a spherical triangle is the point on the
                # sphere equidistant from the three vertices. It can be found by solving
                # the system of equations. A robust way is to use the formula for the
                # circumcenter of a triangle in 3D space, then project to the sphere.
                
                # Let's use the formula: the circumcenter c satisfies:
                # c . (p1 - p2) = 0
                # c . (p1 - p3) = 0
                # |c| = 1
                # This is a linear system for c in the plane orthogonal to the normal of the triangle.
                
                # A simpler approach: the circumcenter is the normalized vector:
                # c = (p1 * |p2|^2 * (p3 - p2) + ...) -- this is getting complicated.
                
                # Let's use a numerical approach: the circumcenter is the point on the sphere
                # that minimizes the maximum distance to the three vertices. But we can compute
                # it as the intersection of the two great circles that are the perpendicular
                # bisectors of the arcs p1-p2 and p1-p3.
                
                # A standard formula for the circumcenter of a spherical triangle with
                # vertices a, b, c (unit vectors) is:
                # c = (a * (b x c) + b * (c x a) + c * (a x b)) / |...|
                # This is the pole of the great circle, but let's verify.
                
                # Actually, the circumcenter is the normalized sum of the unit vectors
                # pointing to the midpoints of the arcs, weighted by the opposite side lengths.
                
                # To save time and complexity, let's use a simple numerical method:
                # The circumcenter is the point on the sphere equidistant from p1, p2, p3.
                # We can find it by solving the linear system:
                # (p1 - p2) . c = 0
                # (p1 - p3) . c = 0
                # |c| = 1
                # Let n = (p1 - p2) x (p1 - p3) be the normal to the plane.
                # The circumcenter lies in the plane of p1, p2, p3, so c = alpha * n + beta * (p1 x p2)
                # This is getting too complicated for a simple implementation.
                
                # Let's use a different approach: the covering radius is the max over all
                # points on the sphere of the distance to the nearest point. The worst points
                # are the circumcenters of the Delaunay triangles. We can compute the
                # circumradius of each triangle by finding the circumcenter numerically.
                
                # For a triangle with vertices a, b, c on the unit sphere, the circumcenter
                # is the point x on the sphere such that x.a = x.b = x.c.
                # This is equivalent to x being in the null space of [a-b; a-c] and |x|=1.
                # Let's compute the null space of the 2x3 matrix [a-b; a-c].
                
                A = np.array([p1 - p2, p1 - p3])
                # The null space of A is spanned by the cross product of the rows
                c_vec = np.cross(A[0], A[1])
                if np.linalg.norm(c_vec) < 1e-10:
                    # Degenerate triangle
                    continue
                c_vec = c_vec / np.linalg.norm(c_vec)
                # Ensure c_vec is on the same side as the triangle (dot product with p1 should be positive)
                if np.dot(c_vec, p1) < 0:
                    c_vec = -c_vec
                # The circumradius is the angle between c_vec and p1
                cos_rad = np.dot(c_vec, p1)
                cos_rad = np.clip(cos_rad, -1.0, 1.0)
                rad = np.arccos(cos_rad)
                if rad > max_rad:
                    max_rad = rad
            return max_rad
        except:
            return np.pi  # Fallback

    # Initial radius
    best_radius = compute_covering_radius(points)
    
    # Iterative improvement
    # We will use a simple heuristic: move each point towards the circumcenter of the
    # worst triangle it belongs to.
    
    # To find which triangles each point belongs to, we can precompute this.
    
    def get_point_triangles(pts):
        try:
            hull = ConvexHull(pts)
            facets = hull.simplices
            point_tris = [[] for _ in range(N)]
            for tri in facets:
                for idx in tri:
                    point_tris[idx].append(tri)
            return point_tris, facets
        except:
            return [[] for _ in range(N)], []

    for iteration in range(1000):
        if time.time() > deadline:
            break
            
        point_tris, facets = get_point_triangles(points)
        if len(facets) == 0:
            break
            
        # Compute circumradii for each facet
        circumradii = []
        circumcenters = []
        for tri in facets:
            p1, p2, p3 = points[tri]
            A = np.array([p1 - p2, p1 - p3])
            c_vec = np.cross(A[0], A[1])
            norm_c = np.linalg.norm(c_vec)
            if norm_c < 1e-10:
                circumradii.append(0.0)
                circumcenters.append(None)
                continue
            c_vec = c_vec / norm_c
            if np.dot(c_vec, p1) < 0:
                c_vec = -c_vec
            cos_rad = np.clip(np.dot(c_vec, p1), -1.0, 1.0)
            rad = np.arccos(cos_rad)
            circumradii.append(rad)
            circumcenters.append(c_vec)
            
        circumradii = np.array(circumradii)
        max_rad = np.max(circumradii)
        
        if max_rad < best_radius:
            best_radius = max_rad
            best_points = points.copy()
            
        # If the covering radius is very small, we can stop
        if max_rad < 0.1:  # ~5.7 degrees
            break
            
        # Move each point towards the circumcenter of the worst triangle it belongs to
        # We will use a weighted move: for each point, find the worst triangle it is in,
        # and move it towards the circumcenter of that triangle.
        
        # To be more stable, we can use a gradient-like step: for each point, compute
        # the gradient of the covering radius with respect to its position.
        
        # A simpler approach: for each point, find the triangle with the largest circumradius
        # that it is part of, and move the point towards the circumcenter of that triangle.
        
        new_points = points.copy()
        for i in range(N):
            if not point_tris[i]:
                continue
            # Find the worst triangle for this point
            worst_tris = []
            for tri_idx in point_tris[i]:
                # tri_idx is the index in the facets list
                if circumcenters[tri_idx] is not None:
                    worst_tris.append((circumradii[tri_idx], tri_idx))
            if not worst_tris:
                continue
            worst_tris.sort(reverse=True)
            worst_rad, worst_tri_idx = worst_tris[0]
            c_vec = circumcenters[worst_tri_idx]
            # Move the point towards c_vec
            # The step size should be proportional to the angle between the point and c_vec
            angle = np.arccos(np.clip(np.dot(points[i], c_vec), -1.0, 1.0))
            step_size = min(0.1, angle * 0.5)  # Damped step
            # Move in the direction of c_vec
            new_points[i] = points[i] + step_size * (c_vec - points[i])
            # Normalize
            new_points[i] = new_points[i] / np.linalg.norm(new_points[i])
            
        points = new_points
        
        # Check for coincident points
        min_dist = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(points[i] - points[j])
                if d < min_dist:
                    min_dist = d
        if min_dist < 1e-4:
            # Add a small random perturbation
            points += np.random.normal(0, 0.01, (N, 3))
            points /= np.linalg.norm(points, axis=1, keepdims=True)
            
    # Final check
    final_radius = compute_covering_radius(points)
    if final_radius < best_radius:
        best_points = points.copy()
        
    return best_points
