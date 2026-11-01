"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time

def construct():
    """
    Finds a configuration of 50 points on the unit sphere minimizing the covering radius.
    Uses a combination of multi-start annealing with geometric gradient information.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    def _hull_and_facets(points):
        """Returns convex hull and list of facet vertex indices. Returns None if failure."""
        try:
            hull = ConvexHull(points)
            facets = hull.simplices
            return hull, facets
        except Exception:
            return None

    def _circumcenter_angular(points, facets):
        """
        For each facet, compute the circumcenter direction on the unit sphere.
        The covering radius is the max angular distance from circumcenter to vertices.
        
        Returns: (max_angular_dist, circumcenters_array)
        """
        max_ang = 0.0
        n_facets = len(facets)
        circumcenters = np.zeros((n_facets, 3))
        
        for i, tri in enumerate(facets):
            a = points[tri[0]]
            b = points[tri[1]]
            c = points[tri[2]]
            
            # Normal to the plane of the triangle
            n = np.cross(b - a, c - a)
            n_norm = np.linalg.norm(n)
            
            if n_norm < 1e-10:
                # Degenerate facet, skip or treat as 0
                continue
                
            n = n / n_norm
            
            # We need to determine which side of the plane the origin is on, 
            # and ensure the circumcenter is on the "outside" of the convex hull.
            # The circumcenter of the spherical triangle is the point on the sphere 
            # furthest from the edges. For a convex hull of points on the unit sphere,
            # the Voronoi vertex (circumcenter) lies in the direction of the outward normal
            # if the origin is inside the hull (which it is, as points are on sphere).
            
            # Actually, the circumcenter of the Euclidean triangle formed by a,b,c 
            # lies on the line through the origin perpendicular to the plane.
            # Let C = lambda * n.
            # Distance from C to a is R.
            # |lambda*n - a|^2 = R^2
            # lambda^2 - 2*lambda*(n.a) + |a|^2 = R^2  (since |n|=1, |a|=1)
            # lambda^2 - 2*lambda*(n.a) + 1 = R^2
            
            # Also for b:
            # lambda^2 - 2*lambda*(n.b) + 1 = R^2
            # Since n is normal to (b-a), n.b = n.a. So this is consistent.
            
            # We have two unknowns: lambda and R.
            # We need another condition. The circumcenter is the center of the circle.
            # The power of the origin with respect to the sphere of radius R centered at C is 1 - R^2.
            # But simpler: The distance from origin to plane is h = |n.a|.
            # The radius of the circle (intersection of plane and unit sphere) is r_circ = sqrt(1 - h^2).
            # Wait, this is the radius of the circle on the sphere? No, this is the Euclidean radius of the circle in the plane.
            
            # Let's find lambda.
            # The circumcenter C is the point in the plane such that |C-a| = |C-b| = |C-c|.
            # We know C is on the line lambda*n.
            # Let's use the property that the projection of the circumcenter onto the plane is the circumcenter of the triangle a,b,c in the plane.
            # The distance from the origin to the plane is h = n.a.
            # The circumcenter of the triangle in the plane, relative to the projection of the origin P = -h*n,
            # Let's just solve for lambda directly.
            # |lambda*n - a|^2 = |lambda*n - b|^2
            # lambda^2 - 2 lambda n.a + 1 = lambda^2 - 2 lambda n.b + 1
            # n.a = n.b (always true)
            
            # We need to use the fact that C is equidistant from a,b,c.
            # Let's use the formula for circumcenter of a triangle in 3D.
            # d = 2 * (a_x (b_y c_z - c_y b_z) + b_x (c_y a_z - a_y c_z) + c_x (a_y b_z - b_y a_z))
            # If d == 0, points are collinear.
            
            d = 2.0 * (
                a[0] * (b[1] * c[2] - c[1] * b[2]) -
                a[1] * (b[0] * c[2] - c[0] * b[2]) +
                a[2] * (b[0] * c[1] - c[0] * b[1])
            )
            
            if abs(d) < 1e-12:
                # Fallback: use centroid direction
                centroid = (a + b + c) / 3.0
                centroid_norm = np.linalg.norm(centroid)
                if centroid_norm > 1e-10:
                    u = centroid / centroid_norm
                    circumcenters[i] = u
                    ang = np.arccos(np.clip(np.dot(u, a), -1.0, 1.0))
                    if ang > max_ang: max_ang = ang
                continue

            # |a|^2 = 1, |b|^2 = 1, |c|^2 = 1
            # U = ( (|b|^2 (c-a) x (c-b) + |c|^2 (a-b) x (a-c) ) / (2 * d) ) ? 
            # Correct formula:
            # U_x = ( (c[1]*a[2]-c[2]*a[1]) + (a[1]*b[2]-a[2]*b[1]) - (b[1]*c[2]-b[2]*c[1]) ) / d  ... wait, weights are |a|^2 etc.
            # Since all are 1:
            U_x = ( (c[1] * a[2] - c[2] * a[1]) + (a[1] * b[2] - a[2] * b[1]) - (b[1] * c[2] - b[2] * c[1]) ) / d
            U_y = ( (a[2] * b[0] - a[0] * b[2]) + (b[2] * c[0] - b[0] * c[2]) - (c[2] * a[0] - c[0] * a[2]) ) / d
            U_z = ( (b[0] * c[1] - b[1] * c[0]) + (c[0] * a[1] - c[1] * a[0]) - (a[0] * b[1] - a[1] * b[0]) ) / d
            
            U = np.array([U_x, U_y, U_z])
            U_norm = np.linalg.norm(U)
            
            if U_norm < 1e-10:
                continue
                
            # The circumcenter on the sphere is U / |U|
            # We must ensure it's the "far" circumcenter (outside the hull relative to origin)
            # The origin is inside the convex hull. The circumcenter of the facet should be on the side away from the origin?
            # Actually, for a point on the sphere, the Voronoi cell is a spherical polygon. The vertex is the point equidistant from the sites.
            # There are two such points (antipodal). We want the one closer to the facet vertices (covering radius < 90 deg usually).
            # If dot(U, a) < 0, we flip it.
            if np.dot(U, a) < 0:
                U = -U
                
            U = U / np.linalg.norm(U)
            circumcenters[i] = U
            
            ang = np.arccos(np.clip(np.dot(U, a), -1.0, 1.0))
            if ang > max_ang:
                max_ang = ang
                
        return max_ang, circumcenters

    def _get_covering_radius(points):
        """Compute covering radius and identify the worst facets."""
        result = _hull_and_facets(points)
        if result is None:
            return np.pi, None, None
        hull, facets = result
        ang, cc = _circumcenter_angular(points, facets)
        return ang, facets, cc

    def _compute_gradient(points, facets, cc):
        """
        Compute gradient of covering radius w.r.t. point positions.
        Only the facet with maximum radius contributes.
        """
        # Find max angle
        angles = np.zeros(len(facets))
        for i in range(len(facets)):
            if np.linalg.norm(cc[i]) < 1e-10: continue
            u = cc[i]
            tri = facets[i]
            # Use vertex 0
            a = points[tri[0]]
            angles[i] = np.arccos(np.clip(np.dot(u, a), -1.0, 1.0))
            
        max_idx = np.argmax(angles)
        if angles[max_idx] == 0: return np.zeros((N, 3))
        
        grad = np.zeros((N, 3))
        tri = facets[max_idx]
        u = cc[max_idx]
        
        # The angular distance is theta = arccos(u . a)
        # d(theta)/d(a) = -u / sqrt(1 - (u.a)^2)
        # But we are moving a on the sphere.
        # The gradient of the radius w.r.t. a is -u (normalized projection).
        # Actually, if we move a, the circumcenter u changes too. This is complex.
        # Approximation: Move the vertices of the worst facet towards the circumcenter.
        
        for k in range(3):
            idx = tri[k]
            p = points[idx]
            # Move p towards u
            # The gradient direction on the sphere is the projection of (u - p)
            move_dir = u - np.dot(u, p) * p
            norm_move = np.linalg.norm(move_dir)
            if norm_move > 1e-10:
                grad[idx] += move_dir / norm_move
                
        return grad

    # Main optimization loop
    best_pts = None
    best_radius = np.pi
    
    # Initialization
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    
    # Multi-start
    num_starts = 5
    for s in range(num_starts):
        if time.time() > deadline - 20: break
        
        if s > 0:
            # Perturb initial points
            noise = np.random.randn(N, 3) * 0.1
            pts = pts + noise
            pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        
        current_pts = pts.copy()
        current_radius, facets, cc = _get_covering_radius(current_pts)
        local_best_pts = current_pts.copy()
        local_best_radius = current_radius
        
        T = 0.1
        T_min = 0.001
        cooling = 0.98
        
        steps = 0
        max_steps = 500
        
        while T > T_min and steps < max_steps:
            if time.time() > deadline - 5: break
            steps += 1
            
            # Gradient step
            grad = _compute_gradient(current_pts, facets, cc)
            
            # Random perturbation
            idx = np.random.randint(N)
            rand_dir = np.random.randn(3)
            rand_dir = rand_dir / np.linalg.norm(rand_dir)
            
            # Combine gradient and random
            # Move all points slightly along gradient, and one point along random
            step_size = 0.01 * T / T_min
            # Actually, let's just do random moves but biased by gradient for the worst facet?
            # Simpler: Pure random move for SA, but use gradient to re-initialize if stuck?
            
            # Let's try: Move a random point in a random direction.
            old_pt = current_pts[idx].copy()
            move = 0.05 * np.sqrt(T) * rand_dir
            new_pt = current_pts[idx] + move
            new_pt = new_pt / np.linalg.norm(new_pt)
            
            current_pts[idx] = new_pt
            
            new_radius, new_facets, new_cc = _get_covering_radius(current_pts)
            
            if new_radius < current_radius or (np.random.random() < np.exp(-(new_radius - current_radius) / T)):
                current_radius = new_radius
                facets = new_facets
                cc = new_cc
                if new_radius < local_best_radius:
                    local_best_radius = new_radius
                    local_best_pts = current_pts.copy()
            else:
                current_pts[idx] = old_pt
                
            T *= cooling
        
        if local_best_radius < best_radius:
            best_radius = local_best_radius
            best_pts = local_best_pts
            # If we found a very good one, we can stop early or refine
            if best_radius < np.radians(18.5):
                # Try to refine the best found so far with a lower temperature
                pass

    if best_pts is None:
        return pts
        
    return best_pts
# EVOLVE-BLOCK-END
