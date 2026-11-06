"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull


def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s eval / 150s kill

    # Initialize with Fibonacci spiral points
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def compute_covering_radius(points):
        """Compute covering radius using convex hull facets."""
        try:
            hull = ConvexHull(points)
            # For each facet, find the circumcenter and radius
            max_radius = 0.0
            for simplex in hull.simplices:
                face_points = points[simplex]
                # Compute circumcenter of the triangle (facet)
                a, b, c = face_points
                # Edge vectors
                ab = b - a
                ac = c - a
                # Normal vector
                normal = np.cross(ab, ac)
                normal_norm = np.linalg.norm(normal)
                if normal_norm < 1e-12:
                    continue
                normal = normal / normal_norm
                # Circumcenter formula for triangle
                # Using the formula: circumcenter = a + (|ac|^2 * (b-a) - |ab|^2 * (c-a)) / (2 * |cross(ab,ac)|)
                # Actually, let's use a more robust approach
                # The circumcenter in 3D for a triangle on a sphere
                # We can use the fact that for points on unit sphere, the circumcenter of the facet
                # is along the normal direction
                # Distance from center to each vertex is 1 (unit sphere)
                # The circumradius R of the spherical triangle satisfies:
                # For a triangle with angles alpha, beta, gamma at vertices,
                # the circumradius is related to the angles.
                
                # Simpler approach: compute the circumcenter of the 3 points in 3D
                # and then find the distance from that circumcenter to the origin
                # Actually, for points on unit sphere, the circumcenter of the facet
                # (the point equidistant from all three vertices in 3D) lies on the line
                # through the origin and the face normal.
                
                # Let's compute the circumcenter properly
                # For triangle ABC on unit sphere, the circumcenter O_c satisfies:
                # |O_c - A| = |O_c - B| = |O_c - C|
                # and O_c = t * normal for some t
                # |t*n - A|^2 = 1 (since A is on unit sphere, but circumcenter is not necessarily on sphere)
                # Actually, we want the angular distance, which is arccos(A · B / |A||B|)
                # The covering radius for a facet is the maximum angular distance from any point
                # in the spherical triangle to the nearest vertex.
                
                # For a spherical triangle, the farthest point from all vertices is the circumcenter
                # of the spherical triangle, which is the point equidistant from all three vertices.
                # This point is along the normal to the facet.
                
                # Let's compute t such that |t*n - A| = |t*n - B| = |t*n - C|
                # |t*n|^2 - 2t(n·A) + |A|^2 = |t*n|^2 - 2t(n·B) + |B|^2
                # -2t(n·A) + 1 = -2t(n·B) + 1
                # n·A = n·B (which is true since n is perpendicular to AB)
                
                # The circumradius in angular distance is arccos(t * n · A) where t is chosen
                # such that the point is equidistant.
                
                # Actually, for points on unit sphere, the spherical circumcenter is the point
                # on the normal direction at distance 1/sin(alpha/2) or something...
                
                # Let me use a different approach: compute the circumcenter of the 3 points
                # in 3D space, then project to unit sphere, then compute angular distances.
                
                # Circumcenter of triangle in 3D:
                d = 2 * (a[0]*(b[1]*c[2]-c[1]*b[2]) - a[1]*(b[0]*c[2]-c[0]*b[2]) + a[2]*(b[0]*c[1]-c[0]*b[1]))
                if abs(d) < 1e-12:
                    continue
                ux = ((a[0]**2+b[0]**2-c[0]**2)*(b[1]*c[2]-c[1]*b[2]) - (a[0]**2+c[0]**2-b[0]**2)*(a[1]*c[2]-c[1]*a[2]) + (b[0]**2+c[0]**2-a[0]**2)*(a[1]*b[2]-b[1]*a[2])) / d
                uy = ((a[1]**2+b[1]**2-c[1]**2)*(b[0]*c[2]-c[0]*b[2]) - (a[1]**2+c[1]**2-b[1]**2)*(a[0]*c[2]-c[0]*a[2]) + (b[1]**2+c[1]**2-a[1]**2)*(a[0]*b[2]-b[0]*a[2])) / d
                uz = ((a[2]**2+b[2]**2-c[2]**2)*(b[0]*c[1]-c[0]*b[1]) - (a[2]**2+c[2]**2-b[2]**2)*(a[0]*c[1]-c[0]*a[1]) + (b[2]**2+c[2]**2-a[2]**2)*(a[0]*b[1]-b[0]*a[1])) / d
                circumcenter = np.array([ux, uy, uz])
                cc_norm = np.linalg.norm(circumcenter)
                if cc_norm < 1e-12:
                    continue
                circumcenter = circumcenter / cc_norm
                # Angular distance from circumcenter to each vertex
                dot_vals = np.clip(np.dot(circumcenter, face_points), -1.0, 1.0)
                angles = np.arccos(dot_vals)
                facet_radius = np.max(angles)
                max_radius = max(max_radius, facet_radius)
            return max_radius
        except Exception:
            return np.pi  # Return large value if hull fails

    best_pts = pts.copy()
    best_radius = compute_covering_radius(best_pts)

    # Gradient-like optimization: move points to reduce covering radius
    lr = 0.1
    for iteration in range(500):
        if time.time() - start_time > time_limit:
            break
        if best_radius < 0.197:  # ~11.28 degrees, close to best known 11.316 degrees
            break

        # Compute forces: for each point, move it away from its Voronoi cell center
        # Approximate by moving each point away from the centroid of its nearest neighbors
        # Simple approach: for each point, compute the direction to reduce local covering radius
        
        # For each point, find the farthest point in its Voronoi cell (approximate)
        # and move towards reducing that distance
        
        # Simple heuristic: move each point slightly away from its nearest neighbors
        # to spread them out, but also allow attraction to fill gaps
        
        # Compute pairwise distances
        dots = np.clip(pts @ pts.T, -1.0, 1.0)
        dists = np.arccos(dots)
        
        # For each point, find its 5 nearest neighbors
        k = 5
        neighbor_idx = np.argsort(dists, axis=1)[:, :k+1]
        neighbor_idx = neighbor_idx[:, 1:]  # Exclude self
        
        # Compute average direction to neighbors
        forces = np.zeros_like(pts)
        for i in range(N):
            neighbors = pts[neighbor_idx[i]]
            # Move away from neighbors
            for j, n_idx in enumerate(neighbor_idx[i]):
                diff = pts[i] - pts[n_idx]
                norm_diff = np.linalg.norm(diff)
                if norm_diff > 1e-10:
                    forces[i] += diff / norm_diff
        
        # Normalize forces
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        force_norms = np.where(force_norms < 1e-10, 1.0, force_norms)
        forces = forces / force_norms
        
        # Update points
        new_pts = pts + lr * forces
        # Renormalize
        new_pts = new_pts / np.linalg.norm(new_pts, axis=1, keepdims=True)
        
        # Check if improved
        new_radius = compute_covering_radius(new_pts)
        if new_radius < best_radius:
            best_radius = new_radius
            best_pts = new_pts
            pts = new_pts
            # Decrease learning rate slowly
            lr *= 0.999
        else:
            # Try smaller step
            lr *= 0.9

    return best_pts
# EVOLVE-BLOCK-END
