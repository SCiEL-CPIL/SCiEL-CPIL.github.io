"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import ConvexHull
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Initial Configuration: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Lloyd's Algorithm (Centroidal Voronoi on Sphere)
    # This minimizes the mean distance, leading to a uniform distribution.
    # We run this for a fixed number of iterations or until time allows.
    def lloyd_step(points):
        # Compute Voronoi cells on the sphere using Convex Hull of points + origin?
        # No, standard Lloyd on sphere:
        # 1. Compute Delaunay triangulation (Convex Hull of points).
        # 2. For each point, its Voronoi cell is the dual of its adjacent facets.
        # 3. The centroid of the Voronoi cell on the sphere is the normalized 
        #    sum of the circumcenters of the adjacent facets, weighted by area?
        # Simpler approximation: The centroid of the Voronoi cell is approximately
        # the normalized sum of the adjacent facet circumcenters.
        
        try:
            hull = ConvexHull(points)
        except Exception:
            return points
            
        # Map each point to its adjacent facets
        point_facets = {i: [] for i in range(N)}
        for f in hull.simplices:
            for idx in f:
                point_facets[idx].append(f)
                
        new_points = np.zeros_like(points)
        for i in range(N):
            if not point_facets[i]:
                new_points[i] = points[i]
                continue
            centroid = np.zeros(3)
            for f in point_facets[i]:
                # Compute circumcenter of the facet
                v1, v2, v3 = points[f[0]], points[f[1]], points[f[2]]
                # Circumcenter of triangle v1, v2, v3 on unit sphere
                # c is proportional to (v1 x v2) x (v2 x v3) ... or solve linear system
                # Let's use the formula for circumcenter of a triangle on sphere
                # c = normalize( (v1 x v2) x (v2 x v3) + ... )?
                # Actually, the circumcenter C satisfies C . v1 = C . v2 = C . v3 = cos(alpha)
                # This is a linear system.
                # Or use the geometric property: C is in the plane of v1, v2, v3.
                # C = a*v1 + b*v2 + c*v3
                # C . v1 = C . v2 = C . v3
                # This can be solved.
                # Simpler: Use the fact that for small triangles, the centroid is a good approx.
                # But for Lloyd, we need the centroid of the Voronoi cell.
                # The Voronoi cell centroid is the normalized sum of the facet circumcenters.
                
                # Calculate circumcenter of triangle (v1, v2, v3)
                # Vector cross products
                e1 = v2 - v1
                e2 = v3 - v1
                n = np.cross(e1, e2)
                n_norm = np.linalg.norm(n)
                if n_norm < 1e-10:
                    continue
                n = n / n_norm
                
                # Circumcenter C on the sphere
                # C = (v1 + v2 + v3) / 3 is the planar centroid, not spherical.
                # The spherical circumcenter is the intersection of the sphere and the line 
                # passing through the planar centroid in the direction of the normal? No.
                # The circumcenter lies in the plane of the triangle.
                # Let C = v1 + u*e1 + v*e2.
                # C . v1 = C . v2 => (v1 + u*e1 + v*e2).v1 = (v1 + u*e1 + v*e2).v2
                # 1 + u(e1.v1) + v(e2.v1) = v1.v2 + u(e1.v2) + v(e2.v2)
                # Similarly for v3.
                # This is a 2x2 linear system for u, v.
                
                A = np.array([
                    [e1.dot(v1) - e1.dot(v2), e2.dot(v1) - e2.dot(v2)],
                    [e1.dot(v1) - e1.dot(v3), e2.dot(v1) - e2.dot(v3)]
                ])
                b = np.array([
                    v1.dot(v2) - 1.0,
                    v1.dot(v3) - 1.0
                ])
                
                try:
                    uv = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    # Degenerate
                    cc = (v1 + v2 + v3) / 3.0
                    cc = cc / np.linalg.norm(cc)
                    centroid += cc
                    continue
                    
                C = v1 + uv[0]*e1 + uv[1]*e2
                C_norm = np.linalg.norm(C)
                if C_norm < 1e-10:
                    continue
                C = C / C_norm
                centroid += C
                
            if np.linalg.norm(centroid) > 1e-10:
                new_points[i] = centroid / np.linalg.norm(centroid)
            else:
                new_points[i] = points[i]
                
        return new_points

    # Run Lloyd's for a while
    for _ in range(100):
        if time.time() > deadline - 30:
            break
        points = lloyd_step(points)

    # 3. Refinement: Reduce Covering Radius
    # The covering radius is the max circumradius of the convex hull facets.
    def compute_circumradius(v1, v2, v3):
        # Returns the angular circumradius of the triangle on the unit sphere
        e1 = v2 - v1
        e2 = v3 - v1
        n = np.cross(e1, e2)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            return 0.0, np.zeros(3)
        n = n / n_norm
        
        A = np.array([
            [e1.dot(v1) - e1.dot(v2), e2.dot(v1) - e2.dot(v2)],
            [e1.dot(v1) - e1.dot(v3), e2.dot(v1) - e2.dot(v3)]
        ])
        b = np.array([
            v1.dot(v2) - 1.0,
            v1.dot(v3) - 1.0
        ])
        
        try:
            uv = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return 0.0, np.zeros(3)
            
        C = v1 + uv[0]*e1 + uv[1]*e2
        C_norm = np.linalg.norm(C)
        if C_norm < 1e-10:
            return 0.0, np.zeros(3)
        C = C / C_norm
        
        # Angular distance from C to v1
        dot = np.clip(np.dot(C, v1), -1.0, 1.0)
        r = np.arccos(dot)
        return r, C

    # Iterative refinement
    for _ in range(200):
        if time.time() > deadline:
            break
        
        try:
            hull = ConvexHull(points)
        except Exception:
            break
            
        max_r = 0.0
        max_facet_idx = -1
        max_circumcenter = None
        
        for idx, f in enumerate(hull.simplices):
            r, C = compute_circumradius(points[f[0]], points[f[1]], points[f[2]])
            if r > max_r:
                max_r = r
                max_facet_idx = idx
                max_circumcenter = C
                
        if max_facet_idx == -1:
            break
            
        # Move the vertices of the worst facet towards its circumcenter
        # to shrink the facet and reduce its circumradius
        f = hull.simplices[max_facet_idx]
        step_size = 0.05 # Small step
        
        for idx in f:
            v = points[idx]
            # Move towards C
            new_v = v + step_size * (max_circumcenter - v)
            new_v = new_v / np.linalg.norm(new_v)
            points[idx] = new_v
            
    return points
# EVOLVE-BLOCK-END
