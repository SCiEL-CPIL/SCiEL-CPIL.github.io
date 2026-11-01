"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
import time
import sys

def construct():
    """
    Finds a configuration of 50 points on the unit sphere minimizing the covering radius.
    Uses Simulated Annealing with Voronoi-based covering radius evaluation.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit
    
    # 1. Initialization: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize
    norms = np.linalg.norm(pts, axis=1)
    pts = pts / norms[:, np.newaxis]
    
    # Pre-compute Delaunay triangulation for the initial configuration
    # We need the Voronoi diagram to compute the covering radius accurately.
    # The covering radius is the maximum distance from a point in the sphere to its nearest site.
    # This occurs at the circumcenters of the Voronoi cells (which are the facets of the convex hull of the sites).
    
    def get_covering_radius(points):
        """
        Computes the covering radius of the set of points on the unit sphere.
        The covering radius is the maximum circumradius of the facets of the convex hull.
        """
        try:
            hull = ConvexHull(points)
            # Facets are triangles (indices of vertices)
            # For each facet, find the circumcenter
            # The circumcenter of a spherical triangle is the normalized vector sum of the vertices? 
            # No, the circumcenter of a Euclidean triangle on the sphere is the point equidistant from the 3 vertices.
            # Since the points are on the unit sphere, the circumcenter of the triangle formed by p1, p2, p3 
            # is along the normal to the plane of the triangle, passing through the origin?
            # Actually, for points on a sphere, the circumcenter of the spherical triangle is the intersection of the sphere 
            # with the line passing through the origin and the circumcenter of the Euclidean triangle.
            
            # A more robust way:
            # The farthest point from the sites in a Voronoi cell is the circumcenter of the corresponding facet.
            # Distance from origin to circumcenter of facet (p1, p2, p3) is R_circ.
            # The angular distance is arccos(1 / R_circ) ? No.
            
            # Let's use the property: 
            # The covering radius is max over all facets f of the angular distance from the facet's circumcenter to any vertex of f.
            # Let C be the circumcenter of the Euclidean triangle formed by the 3 points on the sphere.
            # C lies on the line from the origin to the centroid? No.
            
            # Efficient calculation for a triangle (a,b,c) on unit sphere:
            # The circumcenter u is proportional to a x (b x c) + b x (c x a) + c x (a x b) ?
            # Simpler: The circumcenter of a triangle with vertices a,b,c on the unit sphere is:
            # u = (a * sin(A) + b * sin(B) + c * sin(C)) / |...| ?
            
            # Let's use a standard formula for circumcenter of 3 points in 3D:
            # d = 2 * (a_x * (b_y * c_z - c_y * b_z) + b_x * (c_y * a_z - a_y * c_z) + c_x * (a_y * b_z - b_y * a_z))
            # If d is close to 0, points are collinear (degenerate).
            
            max_radius_deg = 0.0
            facets = hull.simplices
            
            for tri in facets:
                a = points[tri[0]]
                b = points[tri[1]]
                c = points[tri[2]]
                
                # Circumcenter calculation
                # Using the formula: 
                # U = ( |b|^2 (c-a) x (c-b) + |c|^2 (a-b) x (a-c) ) / (2 * (a-b) x (a-c) dot ... )
                # Since |a|=|b|=|c|=1:
                # Numerator = (c-a) x (c-b) + (a-b) x (a-c)
                # Denominator = 2 * | (a-b) x (a-c) |^2 ? No.
                
                # Let's use the explicit formula for circumcenter of triangle (a,b,c):
                # d = 2 * (a_x * (b_y * c_z - c_y * b_z) - a_y * (b_x * c_z - c_x * b_z) + a_z * (b_x * c_y - c_x * b_y))
                # If d == 0, return inf
                # U_x = ( (|b|^2 * (c_y * a_z - c_z * a_y) + |c|^2 * (a_y * b_z - a_z * b_y)) / d
                # ...
                
                # Vectorized approach for all facets is faster, but let's do a loop for clarity and check time.
                # With N=50, number of facets is approx 2*50-4 = 96. This is very fast.
                
                d = 2.0 * (
                    a[0] * (b[1] * c[2] - c[1] * b[2]) -
                    a[1] * (b[0] * c[2] - c[0] * b[2]) +
                    a[2] * (b[0] * c[1] - c[0] * b[1])
                )
                
                if abs(d) < 1e-12:
                    # Degenerate triangle, points nearly collinear.
                    # The covering radius is likely determined by another facet, or this one is "flat".
                    # For a flat triangle on the sphere, the circumcenter is far away.
                    # We can approximate by checking the distance to the centroid.
                    centroid = (a + b + c) / 3.0
                    centroid_norm = np.linalg.norm(centroid)
                    if centroid_norm < 1e-12:
                        continue
                    centroid = centroid / centroid_norm
                    # Distance from centroid to a
                    dist = np.arccos(np.clip(np.dot(centroid, a), -1.0, 1.0))
                    if dist > max_radius_deg:
                        max_radius_deg = dist
                    continue
                
                b_sq = np.dot(b, b)
                c_sq = np.dot(c, c)
                
                U_x = (b_sq * (c[1] * a[2] - c[2] * a[1]) + c_sq * (a[1] * b[2] - a[2] * b[1])) / d
                U_y = (c_sq * (a[0] * b[2] - a[2] * b[0]) + b_sq * (b[0] * c[2] - b[2] * c[0])) / d
                # The circumcenter u is parallel to (a-b) x (a-c).
                u = np.cross(a - b, a - c)
                if np.linalg.norm(u) < 1e-12:
                    # Degenerate case: points are collinear or identical
                    continue
                u = u / np.linalg.norm(u)
                n = np.cross(a - b, a - c)
                n_norm = np.linalg.norm(n)
                if n_norm < 1e-12:
                    continue
                u = n / n_norm
                # We need to determine the sign of u. The circumcenter should be on the same side of the origin as the centroid?
                # Actually, for a triangle on the unit sphere, the circumcenter is the point on the sphere furthest from the edges?
                # No, the circumcenter of the *spherical* triangle is the point equidistant from the vertices.
                # The Euclidean circumcenter of the triangle formed by a,b,c is C_euc.
                # The spherical circumcenter is C_sph = C_euc / |C_euc|.
                
                # Let's compute C_euc properly.
                # C_euc = a + t1 * (b-a) + t2 * (c-a)
                # This is equivalent to finding the intersection of the plane and the line from origin.
                # The line from origin to C_euc is normal to the plane of the triangle.
                # So C_euc = lambda * n.
                # We know |C_euc - a|^2 = |C_euc - b|^2.
                # |lambda n - a|^2 = |lambda n - b|^2
                # lambda^2 |n|^2 - 2 lambda n.a + |a|^2 = lambda^2 |n|^2 - 2 lambda n.b + |b|^2
                # Since |a|=|b|=1:
                # -2 lambda n.a = -2 lambda n.b
                # lambda (n.a - n.b) = 0
                # This implies lambda = 0 or n.a = n.b.
                # n is normal to (a-b), so n.a = n.b is always true!
                # This means the circumcenter of the Euclidean triangle lies on the line defined by n.
                # We need to find the specific lambda.
                # Use the condition that the distance to a is the radius R.
                # R^2 = |lambda n - a|^2 = lambda^2 |n|^2 - 2 lambda n.a + 1
                # Also, the circumcenter is the center of the circle passing through a,b,c.
                # The distance from the origin to the plane is h = |n . a| / |n|.
                # The radius of the circle is R_circ_euc.
                # R_circ_euc^2 = 1 - h^2.
                # Also R_circ_euc^2 = |C_euc - a|^2.
                # This is getting complicated. Let's use a standard library or a simpler check.
                
                # Actually, for the covering radius, we just need the angular distance from the circumcenter to the vertices.
                # Let u be the normalized normal vector.
                # The angular distance from u to a is arccos(u . a).
                # Is u the circumcenter?
                # If the triangle is acute, the circumcenter is in the direction of the normal.
                # If the triangle is obtuse, the circumcenter is in the opposite direction?
                # For points on a sphere, the "circumcenter" of the Voronoi cell is the point on the sphere furthest from the edges of the cell.
                # This point is the normalized vector of the circumcenter of the Euclidean triangle.
                
                # Let's compute the Euclidean circumcenter coordinates explicitly using the cross product formula:
                # C = ( |b|^2 (c-a) x (c-b) + |c|^2 (a-b) x (a-c) ) / (2 * | (a-b) x (a-c) |^2 ) ? No.
                
                # Correct formula for circumcenter of triangle (a,b,c) in 3D:
                # d = 2 * (a_x (b_y c_z - c_y b_z) + b_x (c_y a_z - a_y c_z) + c_x (a_y b_z - b_y a_z))
                # If d is 0, degenerate.
                # A = b_x^2 + b_y^2 + b_z^2
                # B = c_x^2 + c_y^2 + c_z^2
                # U_x = (A (c_y a_z - c_z a_y) + B (a_y b_z - a_z b_y)) / d
                # U_y = (B (a_z b_x - a_x b_z) + A (b_z c_x - b_x c_z)) / d
                # U_z = (A (b_x c_y - b_y c_x) + B (c_x a_y - c_y a_x)) / d
                
                A = b_sq
                B = c_sq
                # Note: |a|^2 = 1, but the formula usually uses |a|^2, |b|^2, |c|^2.
                # Since |a|=1, let's use 1 for A_a.
                A_a = 1.0
                
                U_x = (A_a * (b[1] * c[2] - b[2] * c[1]) + A * (c[1] * a[2] - c[2] * a[1]) - (A_a * (b[1]*a[2]-b[2]*a[1]) + A * (c[1]*b[2]-c[2]*b[1]))) / d 
                # This is wrong. The formula is:
                # U_x = ( (|b|^2 (c_y a_z - c_z a_y) + |c|^2 (a_y b_z - a_z b_y) - |a|^2 (b_y c_z - b_z c_y)) / d )
                
                U_x = (A * (c[1] * a[2] - c[2] * a[1]) + B * (a[1] * b[2] - a[2] * b[1]) - 1.0 * (b[1] * c[2] - b[2] * c[1])) / d
                U_y = (B * (a[2] * b[0] - a[0] * b[2]) + A_a * (b[2] * c[0] - b[0] * c[2]) - A * (c[2] * a[0] - c[0] * a[2])) / d
                U_z = (A_a * (c[0] * b[1] - c[1] * b[0]) + A * (a[0] * c[1] - a[1] * c[0]) - B * (b[0] * a[1] - b[1] * a[0])) / d
                
                U = np.array([U_x, U_y, U_z])
                U_norm = np.linalg.norm(U)
                if U_norm < 1e-12:
                    continue
                U = U / U_norm
                
                # The circumcenter on the sphere is U.
                # The angular distance to vertex a is:
                dist = np.arccos(np.clip(np.dot(U, a), -1.0, 1.0))
                if dist > max_radius_deg:
                    max_radius_deg = dist
            
            return max_radius_deg
        except Exception:
            return np.pi / 2 # Fallback

    current_radius = get_covering_radius(pts)
    best_pts = pts.copy()
    best_radius = current_radius
    
    T = 0.05  # Initial temperature
    T_min = 0.001
    cooling_rate = 0.999
    
    while T > T_min:
        if time.time() > deadline:
            break
            
        # Generate a candidate
        # Randomly pick a point and move it
        idx = np.random.randint(N)
        # Random direction
        rand_vec = np.random.randn(3)
        rand_vec = rand_vec / np.linalg.norm(rand_vec)
        
        # Move the point
        step_size = 0.1 * T / T_min # Adaptive step? Or fixed?
        # Let's use a small fixed step scaled by temperature
        move = 0.05 * np.sqrt(T) * rand_vec
        
        new_pt = pts[idx] + move
        # Project back to sphere
        new_pt = new_pt / np.linalg.norm(new_pt)
        
        # Update points
        old_pt = pts[idx].copy()
        pts[idx] = new_pt
        
        new_radius = get_covering_radius(pts)
        
        # Acceptance criterion
        if new_radius < current_radius or np.random.random() < np.exp(-(new_radius - current_radius) / T):
            current_radius = new_radius
            if new_radius < best_radius:
                best_radius = new_radius
                best_pts = pts.copy()
        else:
            pts[idx] = old_pt
            
        T *= cooling_rate
        
    return best_pts
# EVOLVE-BLOCK-END
