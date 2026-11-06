"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize


def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s eval / 150s kill

    # Initialize with Fibonacci spiral points
    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def compute_covering_radius(points):
        """Compute covering radius using convex hull facets."""
        try:
            hull = ConvexHull(points)
            max_radius = 0.0
            for simplex in hull.simplices:
                face_points = points[simplex]
                a, b, c = face_points
                # Compute circumcenter of the triangle on the unit sphere
                # The circumcenter is the point equidistant from a, b, c on the sphere
                # It lies along the normal to the plane containing a, b, c
                ab = b - a
                ac = c - a
                normal = np.cross(ab, ac)
                normal_norm = np.linalg.norm(normal)
                if normal_norm < 1e-12:
                    continue
                normal = normal / normal_norm
                
                # The circumcenter is at distance t from origin along normal
                # For points on unit sphere, t = 1/sin(angle) where angle is half the circumradius
                # Actually, for a triangle inscribed in a circle of radius R (chord),
                # the circumradius in 3D is R_3d = abc / (4*Area)
                # But we want angular distance.
                
                # Let's compute the circumcenter coordinates directly
                # Using the formula for circumcenter of triangle in 3D
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
            return np.pi

    def get_worst_facet(points):
        """Return the worst facet indices and the covering radius."""
        try:
            hull = ConvexHull(points)
            max_radius = 0.0
            worst_simplex = None
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
                dot_vals = np.clip(np.dot(circumcenter, face_points), -1.0, 1.0)
                angles = np.arccos(dot_vals)
                facet_radius = np.max(angles)
                if facet_radius > max_radius:
                    max_radius = facet_radius
                    worst_simplex = simplex
            return max_radius, worst_simplex
        except Exception:
            return np.pi, None

    def optimize_from_start(pts, time_budget):
        """Run a local optimization phase from the given starting points."""
        start_time_local = time.time()
        current_pts = pts.copy()
        current_radius = compute_covering_radius(current_pts)
        best_pts = current_pts.copy()
        best_radius = current_radius

        # Phase 1: Gradient-like optimization
        lr = 0.05
        for iteration in range(200):
            if time.time() - start_time_local > time_budget:
                break
            if time.time() - start_time > time_limit - 5:
                break
            if best_radius < 0.197:
                break

            # Compute forces: move points away from nearest neighbors to spread out
            dots = np.clip(current_pts @ current_pts.T, -1.0, 1.0)
            dists = np.arccos(dots)
            
            k = 5
            neighbor_idx = np.argsort(dists, axis=1)[:, :k+1]
            neighbor_idx = neighbor_idx[:, 1:]
            
            forces = np.zeros_like(current_pts)
            for i in range(N):
                for n_idx in neighbor_idx[i]:
                    diff = current_pts[i] - current_pts[n_idx]
                    norm_diff = np.linalg.norm(diff)
                    if norm_diff > 1e-10:
                        forces[i] += diff / norm_diff
            
            force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
            force_norms = np.where(force_norms < 1e-10, 1.0, force_norms)
            forces = forces / force_norms
            
            new_pts = current_pts + lr * forces
            new_pts = new_pts / np.linalg.norm(new_pts, axis=1, keepdims=True)
            
            new_radius = compute_covering_radius(new_pts)
            if new_radius < best_radius:
                best_radius = new_radius
                best_pts = new_pts.copy()
                current_pts = new_pts
                lr *= 0.995
            else:
                lr *= 0.9

        # Phase 2: Facet-based refinement (move vertices of worst facet towards circumcenter)
        step_size = 0.03
        stuck_counter = 0
        for iteration in range(500):
            if time.time() - start_time_local > time_budget:
                break
            if time.time() - start_time > time_limit - 5:
                break
            if best_radius < 0.197:
                break

            current_radius, worst_simplex = get_worst_facet(current_pts)
            if worst_simplex is None:
                break

            tri = current_pts[worst_simplex]
            v1, v2, v3 = tri
            c_vec = np.cross(v1 - v2, v2 - v3)
            c_norm = np.linalg.norm(c_vec)
            if c_norm < 1e-9:
                c_vec = (v1 + v2 + v3) / 3.0
                c_norm = np.linalg.norm(c_vec)
            if c_norm < 1e-9:
                break
            c_vec /= c_norm

            new_pts = current_pts.copy()
            for idx in worst_simplex:
                v = new_pts[idx]
                new_v = v + step_size * (c_vec - v)
                new_v /= np.linalg.norm(new_v)
                new_pts[idx] = new_v

            new_radius = compute_covering_radius(new_pts)
            if new_radius < current_radius:
                current_pts = new_pts
                current_radius = new_radius
                if current_radius < best_radius:
                    best_radius = current_radius
                    best_pts = current_pts.copy()
                stuck_counter = 0
            else:
                stuck_counter += 1
                if stuck_counter > 30:
                    step_size *= 0.9
                    stuck_counter = 0
                if step_size < 1e-5:
                    break

        return best_pts, best_radius

    # Multi-restart strategy
    best_pts = None
    best_radius = np.pi

    # Start 1: Fibonacci spiral
    pts1 = fibonacci_spiral(N)
    pts1 = pts1 / np.linalg.norm(pts1, axis=1, keepdims=True)
    result1, rad1 = optimize_from_start(pts1, 30.0)
    if rad1 < best_radius:
        best_radius = rad1
        best_pts = result1

    if time.time() - start_time > time_limit - 10:
        return best_pts

    # Start 2: Perturbed Fibonacci
    pts2 = fibonacci_spiral(N)
    noise = np.random.randn(N, 3) * 0.1
    pts2 = pts2 + noise
    pts2 = pts2 / np.linalg.norm(pts2, axis=1, keepdims=True)
    result2, rad2 = optimize_from_start(pts2, 30.0)
    if rad2 < best_radius:
        best_radius = rad2
        best_pts = result2

    if time.time() - start_time > time_limit - 10:
        return best_pts

    # Start 3: Another perturbation
    pts3 = fibonacci_spiral(N)
    noise = np.random.randn(N, 3) * 0.05
    pts3 = pts3 + noise
    pts3 = pts3 / np.linalg.norm(pts3, axis=1, keepdims=True)
    result3, rad3 = optimize_from_start(pts3, 30.0)
    if rad3 < best_radius:
        best_radius = rad3
        best_pts = result3

    if time.time() - start_time > time_limit - 10:
        return best_pts

    # Final polish on the best result
    if best_pts is not None and time.time() - start_time < time_limit - 15:
        final_result, final_rad = optimize_from_start(best_pts, time_limit - time.time() - 15)
        if final_rad < best_radius:
            best_radius = final_rad
            best_pts = final_result

    return best_pts
# EVOLVE-BLOCK-END
