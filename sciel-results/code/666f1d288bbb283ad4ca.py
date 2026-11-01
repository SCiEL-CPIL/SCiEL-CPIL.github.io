import numpy as np
from scipy.optimize import minimize
import time

N = 130

def construct():
    start_time = time.time()
    
    # 1. Initial guess: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Energy minimization
    # We minimize sum_{i != j} 1/||p_i - p_j||^k + penalty for ||p_i|| != 1
    k = 8
    lambda_penalty = 1000.0
    
    def objective(x):
        p = x.reshape(N, 3)
        # Penalty for not being on the sphere
        norms = np.linalg.norm(p, axis=1)
        penalty = np.sum((norms - 1.0)**2)
        
        # Repulsive energy
        energy = 0.0
        # Vectorized computation of pairwise distances
        # This is O(N^2), which is 130^2 = 16900, very fast.
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Avoid division by zero
        np.fill_diagonal(dists, 1.0)
        energy = np.sum(1.0 / (dists**k))
        
        return energy + lambda_penalty * penalty
    
    # Initial x
    x0 = points.flatten()
    
    # Optimize
    # L-BFGS-B is good for large problems
    res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10})
    
    # Extract and normalize
    points = res.x.reshape(N, 3)
    points = points / np.linalg.norm(points, axis=1)[:, np.newaxis]
    
    # 3. Local refinement: Move worst triangle vertices towards circumcenter
    # This is a simple heuristic to improve covering radius
    for _ in range(50):
        if time.time() - start_time > 110: # Safety margin
            break
            
        # Compute Delaunay triangulation
        from scipy.spatial import Delaunay
        try:
            tri = Delaunay(points)
            # Get hull indices
            hull_idx = tri.convex_hull
            # For each triangle in hull, compute circumcenter and radius
            worst_radius = -1
            worst_tri_idx = -1
            worst_center = None
            worst_verts = None
            
            for t_idx in hull_idx:
                v1, v2, v3 = points[t_idx]
                # Circumcenter of triangle on sphere
                # Use the formula for circumcenter of a triangle in 3D
                # Project onto the plane of the triangle, then find center
                # Simpler: solve for c such that |c-v1|^2 = |c-v2|^2 = |c-v3|^2 and c is on sphere
                # This is equivalent to finding the point on the sphere equidistant to v1, v2, v3
                # which is the intersection of the sphere with the line perpendicular to the triangle plane through its centroid? No.
                # It's the point on the sphere that maximizes the minimum distance to v1, v2, v3? No, it's the circumcenter.
                
                # Let's use the property that the circumcenter c satisfies:
                # c . (v1 - v2) = 0.5 * (|v1|^2 - |v2|^2) = 0
                # c . (v1 - v3) = 0
                # |c|^2 = 1
                # So c is orthogonal to (v1-v2) and (v1-v3).
                # Thus c is parallel to cross(v1-v2, v1-v3).
                normal = np.cross(v1 - v2, v1 - v3)
                if np.linalg.norm(normal) < 1e-10:
                    continue
                c = normal / np.linalg.norm(normal)
                # Check if c is on the correct side (dot product with centroid should be positive if triangle is acute? No, just normalize)
                # Actually, there are two such points: c and -c. We want the one on the sphere that is "outside" the triangle relative to the center?
                # The circumcenter of a spherical triangle is the point on the sphere equidistant to the vertices.
                # It is given by the normalized cross product of the edge vectors?
                # Let's verify.
                # Distance from c to v1: |c - v1|^2 = 1 + 1 - 2 c.v1 = 2(1 - c.v1).
                # We want c.v1 = c.v2 = c.v3.
                # c is parallel to cross(v1-v2, v1-v3).
                # Let's check if c.v1 == c.v2.
                # c . (v1 - v2) = 0 => c.v1 = c.v2. Yes.
                # So c = normalize(cross(v1-v2, v1-v3)) is a circumcenter.
                # But is it the one inside the triangle?
                # The distance is determined by c.v1.
                # We want the circumcenter that is "far" from the center of the sphere? No, it's on the sphere.
                # There are two antipodal points. One is the circumcenter of the triangle, the other is of the antipodal triangle.
                # We want the one that is "close" to the triangle's centroid.
                centroid = (v1 + v2 + v3) / 3.0
                if np.dot(c, centroid) < 0:
                    c = -c
                    
                radius = np.arccos(np.clip(np.dot(c, v1), -1.0, 1.0))
                if radius > worst_radius:
                    worst_radius = radius
                    worst_tri_idx = t_idx
                    worst_center = c
                    worst_verts = [v1, v2, v3]
            
            if worst_tri_idx != -1:
                # Move vertices towards center
                step = 0.01
                new_points = points.copy()
                for idx, v in zip(t_idx, worst_verts):
                    # Move v towards c
                    new_v = v + step * (worst_center - v)
                    new_points[idx] = new_v / np.linalg.norm(new_v)
                points = new_points
                
        except Exception:
            break
            
    return points
