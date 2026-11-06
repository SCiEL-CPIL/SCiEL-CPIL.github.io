"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize


def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s kill

    # Initialize with Fibonacci spiral points
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def compute_covering_radius(points):
        """Compute covering radius using convex hull facets."""
        try:
            hull = ConvexHull(points)
            max_radius = 0.0
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

                dot_vals = np.clip(np.dot(circumcenter, face_points), -1.0, 1.0)
                angles = np.arccos(dot_vals)
                facet_radius = np.max(angles)

                weight = facet_radius / (np.pi / 6.0)

                for idx in simplex:
                    vertex = points[idx]
                    diff = circumcenter - vertex
                    diff_norm = np.linalg.norm(diff)
                    if diff_norm > 1e-10:
                        forces[idx] += weight * diff / diff_norm

            return forces
        except Exception:
            return np.zeros_like(points)

    # Phase 0: Thomson-like energy minimization to maximize minimum pairwise distance
    def thomson_energy(x):
        """Energy that penalizes close points (inverse power law)."""
        p = x.reshape(N, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        p = p / norms
        dots = np.clip(p @ p.T, -1.0, 1.0)
        dists = np.sqrt(np.maximum(0.0, 2.0 * (1.0 - dots)))
        np.fill_diagonal(dists, 1.0)
        energy = np.sum(1.0 / (dists ** 4))
        return energy

    # Run Thomson minimization
    if time.time() - start_time < time_limit - 80:
        x0 = pts.flatten()
        best_energy = np.inf
        best_x = x0.copy()

        for restart in range(3):
            if time.time() - start_time > time_limit - 80:
                break
            if restart == 0:
                x_start = x0.copy()
            else:
                x_start = x0 + 0.1 * np.random.randn(N * 3)
                x_start = x_start.reshape(N, 3)
                x_start = x_start / np.linalg.norm(x_start, axis=1, keepdims=True)
                x_start = x_start.flatten()

            result = minimize(
                thomson_energy,
                x_start,
                method='L-BFGS-B',
                options={'maxiter': 500, 'gtol': 1e-6, 'ftol': 1e-8}
            )

            if result.fun < best_energy:
                best_energy = result.fun
                best_x = result.x.copy()

        pts = best_x.reshape(N, 3)
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)

    # Phase 0.5: Spectral pair-correlation variance minimization
    # This uses a smooth surrogate that promotes uniformity via pair distance statistics
    def spectral_energy(x):
        """
        Smooth energy based on pair correlation variance.
        Minimizes variance of pairwise angular distances, promoting uniformity.
        """
        if time.time() - start_time > time_limit:
            return np.pi
        p = x.reshape(N, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        p = p / norms

        dots = np.clip(p @ p.T, -1.0, 1.0)
        dists = np.arccos(dots)
        np.fill_diagonal(dists, 0.0)

        # Get upper triangle
        iu = np.triu_indices(N, k=1)
        d = dists[iu]

        # Energy: variance of pair distances + penalty for very small distances
        mean_d = np.mean(d)
        var_d = np.var(d)

        # Also add a soft penalty for distances below target (to avoid clustering)
        target_min = 0.2  # ~11.5 degrees
        penalty = np.sum(np.maximum(0, target_min - d) ** 2)

        energy = var_d + 0.1 * penalty
        return energy

    if time.time() - start_time < time_limit - 60:
        x0 = pts.flatten()
        result = minimize(
            spectral_energy,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 300, 'gtol': 1e-7, 'ftol': 1e-10}
        )
        new_pts = result.x.reshape(N, 3)
        new_pts = new_pts / np.linalg.norm(new_pts, axis=1, keepdims=True)

        # Check if spectral optimization improved things
        current_radius = compute_covering_radius(pts)
        new_radius = compute_covering_radius(new_pts)
        if new_radius < current_radius:
            pts = new_pts

    best_pts = pts.copy()
    best_radius = compute_covering_radius(best_pts)

    # Phase 1: L-BFGS optimization on smooth covering radius approximation
    def smooth_covering_radius(x):
        """
        Smooth approximation of covering radius for L-BFGS.
        """
        if time.time() - start_time > time_limit:
            return np.pi
        p = x.reshape(N, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        p = p / norms

        try:
            hull = ConvexHull(p)
            facet_radii = []
            for simplex in hull.simplices:
                face_points = p[simplex]
                a, b, c = face_points
                ab = b - a
                ac = c - a
                normal = np.cross(ab, ac)
                normal_norm = np.linalg.norm(normal)
                if normal_norm < 1e-12:
                    continue

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

                dot_vals = np.clip(np.dot(circumcenter, face_points), -1.0, 1.0)
                angles = np.arccos(dot_vals)
                facet_radius = np.max(angles)
                facet_radii.append(facet_radius)

            if not facet_radii:
                return np.pi

            facet_radii = np.array(facet_radii)
            beta = 50.0
            max_r = np.max(facet_radii)
            soft_max = max_r + np.log(np.sum(np.exp(beta * (facet_radii - max_r)))) / beta
            return soft_max
        except Exception:
            return np.pi

    if time.time() - start_time < time_limit - 50:
        x0 = pts.flatten()
        result = minimize(
            smooth_covering_radius,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 200, 'gtol': 1e-7, 'ftol': 1e-10}
        )

        if result.fun < best_radius:
            new_pts = result.x.reshape(N, 3)
            new_pts = new_pts / np.linalg.norm(new_pts, axis=1, keepdims=True)
            new_radius = compute_covering_radius(new_pts)
            if new_radius < best_radius:
                best_radius = new_radius
                best_pts = new_pts
                pts = new_pts

    # Phase 2: Facet-based local search (coarse)
    lr = 0.05
    for iteration in range(800):
        if time.time() - start_time > time_limit:
            break
        if best_radius < 0.1972:
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
            lr *= 0.995
        else:
            lr *= 0.85

    # Phase 3: Fine polish
    lr = 0.01
    for iteration in range(400):
        if time.time() - start_time > time_limit:
            break
        if best_radius < 0.1971:
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

    # Phase 4: Very fine polish
    lr = 0.002
    for iteration in range(300):
        if time.time() - start_time > time_limit:
            break
        if best_radius < 0.19705:
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

    # Phase 5: Final high-precision polish with very small steps
    lr = 0.0005
    for iteration in range(200):
        if time.time() - start_time > time_limit - 5:
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
        else:
            break

    return best_pts
# EVOLVE-BLOCK-END
