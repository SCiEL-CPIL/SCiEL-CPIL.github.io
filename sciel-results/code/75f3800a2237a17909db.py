import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

def construct():
    def random_points(n):
        # Uniform random points on sphere
        u = np.random.rand(n)
        v = np.random.rand(n)
        theta = 2 * np.pi * u
        phi = np.arccos(2 * v - 1)
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        return np.column_stack([x, y, z])

    def compute_circumcenters(P, hull):
        """
        Compute circumcenters and circumradii for each face of the convex hull.
        Returns:
        centers: (F, 3) array of circumcenters on the sphere
        radii: (F,) array of circumradii (angular distances)
        """
        faces = hull.simplices
        F = len(faces)
        centers = np.zeros((F, 3))
        radii = np.zeros(F)
        
        for idx, face in enumerate(faces):
            p1 = P[face[0]]
            p2 = P[face[1]]
            p3 = P[face[2]]
            
            # Circumcenter of triangle p1, p2, p3 on the sphere
            # The circumcenter is the point C such that |C-p1| = |C-p2| = |C-p3|
            # C is along the normal to the plane p1,p2,p3.
            # Normal vector n = (p2-p1) x (p3-p1)
            n = np.cross(p2 - p1, p3 - p1)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-10:
                # Degenerate face, skip or handle
                centers[idx] = p1
                radii[idx] = 0
                continue
            n = n / n_norm
            
            # The circumcenter C = k * n.
            # |C - p1|^2 = |C|^2 - 2 C.p1 + |p1|^2 = 1 - 2 k n.p1 + 1 = 2 - 2 k (n.p1)
            # We want to minimize this? No, C is on the sphere, so |C|=1.
            # The condition is that C is equidistant from p1, p2, p3.
            # Actually, the circumcenter of a spherical triangle is the intersection of the sphere with the line passing through the origin and perpendicular to the plane of the triangle?
            # No, that's the center of the circle containing the triangle, but the center of the *great circle* is the normal.
            # The circumcenter of the *spherical* triangle is the point on the sphere that is equidistant from the vertices.
            # This point is indeed along the normal vector.
            # Let C = n. Distance from n to p1 is arccos(n.p1).
            # Is n equidistant? Yes, because n is perpendicular to the plane, so n.p1 = n.p2 = n.p3.
            # So the circumcenter is simply the normalized normal vector.
            # However, we must ensure it's the correct one (the one inside the triangle if it's small, or the antipode).
            # For small triangles, the circumcenter is close to the triangle.
            # The distance is arccos(n.p1).
            
            dot_val = np.dot(n, p1)
            dot_val = np.clip(dot_val, -1.0, 1.0)
            radii[idx] = np.arccos(dot_val)
            centers[idx] = n
            
            # Check if the center is "inside" the triangle.
            # If the triangle is large (> hemisphere), the circumcenter might be on the other side.
            # We want the covering radius to be the max distance to the nearest point.
            # The "empty cap" center is the circumcenter.
            # If the triangle is obtuse, the circumcenter is outside.
            # The covering radius is determined by the largest empty cap.
            # The center of the largest empty cap is either a circumcenter of a Delaunay triangle or a point at infinity (if points are in a hemisphere, but we assume good distribution).
            
        return centers, radii

    def covering_radius(P):
        try:
            hull = ConvexHull(P)
        except Exception:
            return 1.0 # Invalid or degenerate, return large value
            
        centers, radii = compute_circumcenters(P, hull)
        # The covering radius is the maximum circumradius
        return np.max(radii)

    def optimize_local(P):
        """
        Voronoi-based local search.
        """
        P = P.copy()
        alpha = 0.05
        for it in range(200):
            try:
                hull = ConvexHull(P)
            except Exception:
                # If hull fails, perturb
                P += np.random.normal(0, 0.1, P.shape)
                P /= np.linalg.norm(P, axis=1, keepdims=True)
                continue
                
            centers, radii = compute_circumcenters(P, hull)
            faces = hull.simplices
            
            # Map point index to list of face indices
            point_faces = [[] for _ in range(N)]
            for f_idx, face in enumerate(faces):
                for p_idx in face:
                    point_faces[p_idx].append(f_idx)
            
            # Calculate update for each point
            updates = np.zeros_like(P)
            for i in range(N):
                if not point_faces[i]:
                    continue
                    
                # Find the circumcenter of the faces connected to i that is farthest from P[i]
                max_dist = -1
                best_center = None
                for f_idx in point_faces[i]:
                    c = centers[f_idx]
                    # Angular distance
                    dot = np.dot(P[i], c)
                    dot = np.clip(dot, -1.0, 1.0)
                    dist = np.arccos(dot)
                    if dist > max_dist:
                        max_dist = dist
                        best_center = c
                        
                if best_center is not None:
                    # Move P[i] towards best_center
                    # Project best_center onto tangent plane at P[i]
                    proj = best_center - P[i] * np.dot(P[i], best_center)
                    updates[i] = proj / (np.linalg.norm(proj) + 1e-10)
            
            # Apply update
            P = P + alpha * updates
            P /= np.linalg.norm(P, axis=1, keepdims=True)
            
            # Decay step size
            alpha *= 0.95
            
        return P

    # Main loop with restarts
    best_P = None
    best_radius = np.inf
    
    for restart in range(10):
        # Initialize
        if restart == 0:
            # Fibonacci for first try? No, directive says from scratch.
            # Let's use a mix: Random for restart > 0, maybe a structured init for 0?
            # Let's just use random for all to be consistent with "from scratch"
            P = random_points(N)
        else:
            P = random_points(N)
            
        # Optimize
        P_opt = optimize_local(P)
        
        # Evaluate
        radius = covering_radius(P_opt)
        
        if radius < best_radius:
            best_radius = radius
            best_P = P_opt.copy()
            
        # Early stop if we get very good
        if best_radius < np.deg2rad(11.32):
            break

    return best_P
