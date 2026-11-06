"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull


def construct():
    start_time = time.time()
    time_limit = 118.0  # Safety margin for 120s eval / 150s kill

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
            max_radius = 0.0
            for simplex in hull.simplices:
                face_points = points[simplex]
                a, b, c = face_points
                # Compute circumcenter of triangle in 3D
                ab = b - a
                ac = c - a
                normal = np.cross(ab, ac)
                normal_norm = np.linalg.norm(normal)
                if normal_norm < 1e-12:
                    continue
                normal = normal / normal_norm
                
                # For points on unit sphere, the circumcenter of the spherical triangle
                # (point equidistant from all three vertices in angular distance) lies along the normal
                # The angular distance from circumcenter to each vertex is the same
                # We need to find the point on the normal that is equidistant from a, b, c
                
                # The circumcenter in 3D space (not on sphere) for triangle abc:
                # Using the formula with cross products
                d = 2.0 * (a[0] * (b[1] * c[2] - c[1] * b[2]) - a[1] * (b[0] * c[2] - c[0] * b[2]) + a[2] * (b[0] * c[1] - c[0] * b[1]))
                if abs(d) < 1e-12:
                    continue
                
                a2 = np.dot(a, a)
                b2 = np.dot(b, b)
                c2 = np.dot(c, c)
                
                ux = ((a2 * (b[1] * c[2] - c[1] * b[2]) - b2 * (a[1] * c[2] - c[1] * a[2]) + c2 * (a[1] * b[2] - b[1] * a[2])) / d)
                uy = ((a2 * (b[0] * c[2] - c[0] * b[2]) - b2 * (a[0] * c[2] - c[0] * a[2]) + c2 * (a[0] * b[2] - b[0] * a[2])) / d)
                uz = ((a2 * (b[0] * c[1] - c[0] * b[1]) - b2 * (a[0] * c[1] - c[0] * a[1]) + c2 * (a[0] * b[1] - b[0] * a[1])) / d)
                
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
            return np.pi

    def compute_facet_forces(points):
        """
        Compute forces on each vertex based on the facet circumcenters.
        For each facet, the circumcenter is the point farthest from all three vertices.
        We want to move vertices toward their circumcenters to reduce the covering radius.
        """
        try:
            hull = ConvexHull(points)
            forces = np.zeros_like(points)
            
            for simplex in hull.simplices:
                face_points = points[simplex]
                a, b, c = face_points
                ab = b - a
                ac = c - a
                normal = np.cross(ab, ac)
                normal_norm = np.linalg.norm(normal)
                if normal_norm < 1e-12:
                    continue
                normal = normal / normal_norm
                
                d = 2.0 * (a[0] * (b[1] * c[2] - c[1] * b[2]) - a[1] * (b[0] * c[2] - c[0] * b[2]) + a[2] * (b[0] * c[1] - c[0] * b[1]))
                if abs(d) < 1e-12:
                    continue
                
                a2 = np.dot(a, a)
                b2 = np.dot(b, b)
                c2 = np.dot(c, c)
                
                ux = ((a2 * (b[1] * c[2] - c[1] * b[2]) - b2 * (a[1] * c[2] - c[1] * a[2]) + c2 * (a[1] * b[2] - b[1] * a[2])) / d)
                uy = ((a2 * (b[0] * c[2] - c[0] * b[2]) - b2 * (a[0] * c[2] - c[0] * a[2]) + c2 * (a[0] * b[2] - b[0] * a[2])) / d)
                uz = ((a2 * (b[0] * c[1] - c[0] * b[1]) - b2 * (a[0] * c[1] - c[0] * a[1]) + c2 * (a[0] * b[1] - b[0] * a[1])) / d)
                
                circumcenter = np.array([ux, uy, uz])
                cc_norm = np.linalg.norm(circumcenter)
                if cc_norm < 1e-12:
                    continue
                circumcenter = circumcenter / cc_norm
                
                # Angular distance from circumcenter to each vertex
                dot_vals = np.clip(np.dot(circumcenter, face_points), -1.0, 1.0)
                angles = np.arccos(dot_vals)
                facet_radius = np.max(angles)
                
                # Move each vertex toward the circumcenter
                # Weight by the facet radius (larger facets get more force)
                weight = facet_radius / (np.pi / 6.0)  # Normalize by typical radius
                
                for idx in simplex:
                    vertex = points[idx]
                    # Direction from vertex to circumcenter
                    diff = circumcenter - vertex
                    diff_norm = np.linalg.norm(diff)
                    if diff_norm > 1e-10:
                        forces[idx] += weight * diff / diff_norm
            
            return forces
        except Exception:
            return np.zeros_like(points)

    best_pts = pts.copy()
    best_radius = compute_covering_radius(best_pts)

    # Phase 1: Coarse optimization with facet-based forces
    lr = 0.05
    for iteration in range(800):
        if time.time() - start_time > time_limit:
            break
        if best_radius < 0.1972:  # ~11.29 degrees
            break

        forces = compute_facet_forces(pts)
        
        # Normalize forces
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        force_norms = np.where(force_norms < 1e-10, 1.0, force_norms)
        forces = forces / force_norms
        
        # Update points
        new_pts = pts + lr * forces
        new_pts = new_pts / np.linalg.norm(new_pts, axis=1, keepdims=True)
        
        new_radius = compute_covering_radius(new_pts)
        if new_radius < best_radius:
            best_radius = new_radius
            best_pts = new_pts
            pts = new_pts
            lr *= 0.995
        else:
            lr *= 0.85

    # Phase 2: Fine polish with smaller steps
    lr = 0.01
    for iteration in range(400):
        if time.time() - start_time > time_limit:
            break
        if best_radius < 0.1971:  # ~11.285 degrees
            break

        forces = compute_facet_forces(pts)
        
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        force_norms = np.where(force_norms < 1e-10, 1.0, force_norms)
        forces = forces / force_norms
        
        new_pts = pts + lr * forces
        new_pts = new_pts / np.linalg.norm(new_pts, axis=1, keepdims=True)
        
        new_radius = compute_covering_radius(new_pts)
        if new_radius < best_radius:
            best_radius = new_radius
            best_pts = new_pts
            pts = new_pts
            lr *= 0.998
        else:
            lr *= 0.9

    # Phase 3: Very fine polish
    lr = 0.002
    for iteration in range(200):
        if time.time() - start_time > time_limit:
            break
        if best_radius < 0.19705:  # ~11.282 degrees
            break

        forces = compute_facet_forces(pts)
        
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        force_norms = np.where(force_norms < 1e-10, 1.0, force_norms)
        forces = forces / force_norms
        
        new_pts = pts + lr * forces
        new_pts = new_pts / np.linalg.norm(new_pts, axis=1, keepdims=True)
        
        new_radius = compute_covering_radius(new_pts)
        if new_radius < best_radius:
            best_radius = new_radius
            best_pts = new_pts
            pts = new_pts
            lr *= 0.999
        else:
            lr *= 0.95

    return best_pts
# EVOLVE-BLOCK-END
