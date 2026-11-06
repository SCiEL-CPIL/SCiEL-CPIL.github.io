import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

N = 130

# EVOLVE-BLOCK-START
def construct():
    # 1. Initial guess: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    pts = np.column_stack([x, y, z])
    pts = pts / np.linalg.norm(pts, axis=1)[:, np.newaxis]
    
    # 2. Geometric Refinement: Move points towards farthest Voronoi vertices
    # We iterate a few times to get close to a local optimum for the covering radius
    # The Voronoi vertices on the sphere are the circumcenters of the Delaunay triangles (faces of the convex hull)
    
    def get_voronoi_vertices_and_distances(points):
        """
        Computes the circumcenters of the convex hull faces (Delaunay triangles).
        Returns a list of (point_index, circumcenter, distance) for the farthest vertex per point.
        """
        try:
            hull = ConvexHull(points)
        except Exception:
            # If hull fails (e.g., coplanar or numerical issues), return zeros
            return [0.0] * N, np.zeros((N, 3))
        
        # Map face vertices to point indices
        # hull.simplices contains the indices of the vertices for each face
        # We need to know, for each point, which faces are adjacent to it.
        
        # Create an adjacency list: point_idx -> list of face indices
        point_to_faces = {i: [] for i in range(N)}
        for face_idx, simplex in enumerate(hull.simplices):
            for v in simplex:
                point_to_faces[v].append(face_idx)
        
        farthest_dists = np.zeros(N)
        farthest_centers = np.zeros((N, 3))
        
        # Precompute circumcenters for all faces
        # For a triangle on the unit sphere with vertices a, b, c:
        # The circumcenter is in the direction of a+b+c if the triangle is acute.
        # More generally, the circumcenter is the intersection of the sphere and the plane defined by a,b,c.
        # The vector n = (b-a) x (c-a) is normal to the plane.
        # The circumcenter c_c is of the form:
        # c_c = a + s * (b-a) + t * (c-a)
        # It must satisfy |c_c| = 1.
        # A simpler formula for the circumcenter of a spherical triangle (or planar triangle projected):
        # If the triangle is not too large, the circumcenter is roughly (a+b+c) / |a+b+c|.
        # Let's use the exact formula for the circumcenter of a triangle in 3D space.
        # The circumcenter is equidistant from a, b, c.
        # Let C be the circumcenter. C . (b-a) = 0.5 * (|b|^2 - |a|^2) = 0 (since on unit sphere).
        # So C is orthogonal to (b-a) and (c-a). Thus C is parallel to (b-a) x (c-a).
        # C = k * n, where n = (b-a) x (c-a).
        # We need |C| = 1. So C = n / |n| OR C = -n / |n|.
        # We choose the sign such that C is on the same side of the plane as the origin? No, the origin is the center of the sphere.
        # The circumcenter of the triangle on the sphere is the point on the sphere closest to the "center" of the triangle?
        # Actually, for a Delaunay triangle on the sphere, the circumcenter is the point on the sphere such that the circle passing through the 3 vertices has its center there.
        # The vector n = (b-a) x (c-a) points away from the origin if the orientation is consistent.
        # Let's check the dot product with a. If n . a > 0, the center is in the direction n.
        
        face_centers = np.zeros((len(hull.simplices), 3))
        for face_idx, simplex in enumerate(hull.simplices):
            a = points[simplex[0]]
            b = points[simplex[1]]
            c = points[simplex[2]]
            
            n = np.cross(b - a, c - a)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-10:
                face_centers[face_idx] = a # Degenerate
            else:
                n = n / norm_n
                # Determine sign
                # The circumcenter should be "inside" the triangle's cap.
                # If we take the average of the vertices, it points towards the interior.
                # Let's check dot(n, a+b+c). If positive, keep n, else flip.
                if np.dot(n, a + b + c) < 0:
                    n = -n
                face_centers[face_idx] = n
        
        # For each point, find the farthest face center
        for i in range(N):
            faces = point_to_faces[i]
            if not faces:
                continue
            p = points[i]
            min_dist = -1.0
            best_center = p
            for f_idx in faces:
                c = face_centers[f_idx]
                # Angular distance
                # d = arccos(p . c)
                # We want to maximize this distance.
                # Since arccos is monotonic, we can minimize the dot product.
                dot = np.dot(p, c)
                # Clamp to [-1, 1]
                dot = np.clip(dot, -1.0, 1.0)
                if dot < min_dist:
                    min_dist = dot
                    best_center = c
            
            # The farthest center corresponds to the minimum dot product.
            # Distance is arccos(min_dist).
            # We store the center to move towards.
            # To minimize the distance, we should move TOWARDS the center?
            # No, if the center is far, the point is "bad".
            # If we move the point towards the center, the distance decreases.
            # So we move p towards best_center.
            
            # Store the "gradient" direction
            # Direction = best_center - p
            # We will update p += step * (best_center - p)
            
            # For the objective function later, we need the distance.
            # dist = arccos(min_dist)
            farthest_dists[i] = np.arccos(min_dist)
            farthest_centers[i] = best_center
            
        return farthest_dists, farthest_centers

    # Iterative Refinement
    # We perform a few steps of moving points towards their farthest Voronoi vertices
    # This is a form of Lloyd's algorithm for covering.
    step_size = 0.1
    for _ in range(50):
        dists, centers = get_voronoi_vertices_and_distances(pts)
        
        # Update points
        # Move each point towards its farthest center
        # To avoid overshooting, use a small step
        deltas = centers - pts
        # Normalize deltas to unit length to control step size
        norms = np.linalg.norm(deltas, axis=1)
        norms[norms < 1e-10] = 1.0
        deltas = deltas / norms[:, np.newaxis]
        
        pts = pts + step_size * deltas
        pts = pts / np.linalg.norm(pts, axis=1)[:, np.newaxis]
        
        # Check for convergence or stagnation
        # If the max distance doesn't change, break
        # (Simplified: just run fixed iterations)

    # 3. Final Polish with L-BFGS-B
    # Objective: Minimize the sum of exponentials of the local covering radii
    # This smooths the max operation.
    
    def energy(params):
        # params are Cartesian coordinates
        p = params.reshape(N, 3)
        # Normalize
        p = p / np.linalg.norm(p, axis=1)[:, np.newaxis]
        
        # Compute local covering radii (approximate)
        # Re-using the logic from above is expensive.
        # Instead, we use a proxy: for each point, the distance to the 3 nearest neighbors?
        # No, that's packing.
        # For covering, the "worst" point is the one with the largest distance to the nearest neighbor?
        # No, the covering radius is the max over all points y on the sphere.
        # The worst y is a Voronoi vertex.
        # So we really need the Voronoi vertices.
        
        # To make this differentiable, we can use the fact that the Voronoi vertices are determined by the points.
        # But ConvexHull is not differentiable.
        
        # Alternative: Use the distance to the nearest 3 neighbors as a proxy?
        # Or simply use the max distance to the nearest neighbor (min-max problem).
        # Let's use the sum of exp(dist_to_nearest) / T.
        # This penalizes points that are isolated (large distance to nearest neighbor).
        # This is a good proxy for covering radius.
        
        # Compute pairwise distances
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dist_sq, 1.0)
        
        # Find nearest neighbor distance for each point
        min_dist_sq = np.min(dist_sq, axis=1)
        min_dist = np.sqrt(min_dist_sq)
        
        # Minimize sum of exp(min_dist / T)
        T = 0.1
        return np.sum(np.exp(min_dist / T))
    
    # Initial guess for optimization is the refined pts
    x0 = pts.flatten()
    
    result = minimize(energy, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
    
    pts_opt = result.x.reshape(N, 3)
    pts_opt = pts_opt / np.linalg.norm(pts_opt, axis=1)[:, np.newaxis]
    
    return pts_opt
# EVOLVE-BLOCK-END
