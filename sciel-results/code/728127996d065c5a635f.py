"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Leave 15s margin for safety

    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def compute_covering_radius(pts):
        """Compute covering radius as max circumradius over convex hull faces (vectorized)."""
        try:
            hull = ConvexHull(pts)
            faces = hull.simplices  # shape (F, 3)
            a = pts[faces[:, 0]]
            b = pts[faces[:, 1]]
            c = pts[faces[:, 2]]

            ab = np.cross(a, b)
            bc = np.cross(b, c)
            ca = np.cross(c, a)
            c_dir = ab + bc + ca
            norms = np.linalg.norm(c_dir, axis=1)

            radii = np.zeros(len(faces))
            valid = norms > 1e-10

            if np.any(valid):
                c_dir_valid = c_dir[valid] / norms[valid, np.newaxis]
                cos_r = np.clip(np.sum(c_dir_valid * a[valid], axis=1), -1.0, 1.0)
                r = np.arccos(cos_r)
                r[r > np.pi / 2] = np.pi - r[r > np.pi / 2]
                radii[valid] = r

            if np.any(~valid):
                a_deg = a[~valid]
                b_deg = b[~valid]
                c_deg = c[~valid]
                d_ab = np.arccos(np.clip(np.sum(a_deg * b_deg, axis=1), -1, 1))
                d_bc = np.arccos(np.clip(np.sum(b_deg * c_deg, axis=1), -1, 1))
                d_ca = np.arccos(np.clip(np.sum(c_deg * a_deg, axis=1), -1, 1))
                radii[~valid] = np.maximum(np.maximum(d_ab, d_bc), d_ca)

            return float(np.max(radii))
        except Exception:
            # Fallback: grid-based estimate
            lat = np.linspace(-np.pi/2, np.pi/2, 50)
            lon = np.linspace(0, 2*np.pi, 100, endpoint=False)
            LA, LO = np.meshgrid(lat, lon, indexing='ij')
            z = np.sin(LA)
            x = np.cos(LA) * np.cos(LO)
            y = np.cos(LA) * np.sin(LO)
            grid_pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
            dot = np.dot(grid_pts, pts.T)
            dot = np.clip(dot, -1.0, 1.0)
            ang_dists = np.arccos(dot)
            min_dists = np.min(ang_dists, axis=1)
            return float(np.max(min_dists))

    def compute_face_radii_and_circumcenters(pts):
        """Return arrays of face circumradii, circumcenters, and faces for gradient-aware optimization."""
        try:
            hull = ConvexHull(pts)
            faces = hull.simplices
            a = pts[faces[:, 0]]
            b = pts[faces[:, 1]]
            c = pts[faces[:, 2]]

            ab = np.cross(a, b)
            bc = np.cross(b, c)
            ca = np.cross(c, a)
            c_dir = ab + bc + ca
            norms = np.linalg.norm(c_dir, axis=1)

            radii = np.zeros(len(faces))
            circumcenters = np.zeros((len(faces), 3))
            valid = norms > 1e-10

            if np.any(valid):
                c_dir_valid = c_dir[valid] / norms[valid, np.newaxis]
                cos_r = np.clip(np.sum(c_dir_valid * a[valid], axis=1), -1.0, 1.0)
                r = np.arccos(cos_r)
                r[r > np.pi / 2] = np.pi - r[r > np.pi / 2]
                radii[valid] = r
                circumcenters[valid] = c_dir_valid

            if np.any(~valid):
                a_deg = a[~valid]
                b_deg = b[~valid]
                c_deg = c[~valid]
                d_ab = np.arccos(np.clip(np.sum(a_deg * b_deg, axis=1), -1, 1))
                d_bc = np.arccos(np.clip(np.sum(b_deg * c_deg, axis=1), -1, 1))
                d_ca = np.arccos(np.clip(np.sum(c_deg * a_deg, axis=1), -1, 1))
                radii[~valid] = np.maximum(np.maximum(d_ab, d_bc), d_ca)
                # For degenerate faces, use centroid as pseudo-circumcenter
                circumcenters[~valid] = (a_deg + b_deg + c_deg) / 3.0
                cc_norms = np.linalg.norm(circumcenters[~valid], axis=1, keepdims=True)
                circumcenters[~valid] = circumcenters[~valid] / np.where(cc_norms > 1e-10, cc_norms, 1.0)

            return radii, circumcenters, faces
        except Exception:
            return None, None, None

    def facet_gradient_step(points, alpha, face_radii, circumcenters, faces):
        """Move each point toward the circumcenter of its largest adjacent facet (the 'hole' it should fill)."""
        n_pts = len(points)
        # For each vertex, find the largest face it belongs to and move toward that face's circumcenter
        vertex_max_rad = np.zeros(n_pts)
        vertex_best_cc = np.zeros((n_pts, 3))

        for fi in range(len(faces)):
            r = face_radii[fi]
            cc = circumcenters[fi]
            for vi in faces[fi]:
                if r > vertex_max_rad[vi]:
                    vertex_max_rad[vi] = r
                    vertex_best_cc[vi] = cc

        # Direction: from point toward best circumcenter, projected tangent to sphere
        forces = np.zeros((n_pts, 3))
        for i in range(n_pts):
            if vertex_max_rad[i] > 1e-10:
                # Tangent direction from point to circumcenter
                to_cc = vertex_best_cc[i]
                # Project to tangent plane: remove radial component
                radial = np.dot(points[i], to_cc)
                tangent = to_cc - radial * points[i]
                t_norm = np.linalg.norm(tangent)
                if t_norm > 1e-10:
                    tangent /= t_norm
                    # Weight by how large this face is relative to max
                    forces[i] = tangent * (vertex_max_rad[i] / (np.max(face_radii) + 1e-10))

        points = points + alpha * forces
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        return points

    def repulsion_step_density_aware(points, alpha, face_radii=None, faces=None):
        """Repulsion with density-aware weighting: points in large-cap regions get stronger push."""
        dot = np.dot(points, points.T)
        dot = np.clip(dot, -1.0, 1.0)
        np.fill_diagonal(dot, 1.0)
        dist_sq = 2.0 - 2.0 * dot
        dist_sq[dist_sq < 1e-10] = 1e-10
        w = 1.0 / (dist_sq ** 1.5)
        np.fill_diagonal(w, 0.0)

        # Density-aware weighting: if we have face radii and faces, weight vertices in large faces more
        if face_radii is not None and faces is not None:
            vertex_max_rad = np.zeros(len(points))
            for fi in range(len(faces)):
                for vi in faces[fi]:
                    vertex_max_rad[vi] = max(vertex_max_rad[vi], face_radii[fi])
            if np.max(vertex_max_rad) > 1e-10:
                weight_boost = 1.0 + 0.3 * vertex_max_rad / np.max(vertex_max_rad)
                w = w * weight_boost[:, np.newaxis]

        sum_w = np.sum(w, axis=1)
        sum_p_w = w @ points
        forces = points * sum_w[:, np.newaxis] - sum_p_w
        points = points + alpha * forces
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        return points

    def combined_step(points, alpha, face_radii, circumcenters, faces):
        """Combine facet gradient and repulsion for robust optimization."""
        # Apply facet gradient (targeted hole-filling)
        points = facet_gradient_step(points, alpha * 0.7, face_radii, circumcenters, faces)
        # Apply repulsion (prevents clustering)
        points = repulsion_step_density_aware(points, alpha * 0.5, face_radii, faces)
        return points

    def polish_step(points, alpha, n_iter=50):
        """High-precision polish with very small steps and periodic face-aware adjustment."""
        for it in range(n_iter):
            # Every 10 iterations, do a face-aware polish
            if it % 10 == 5:
                face_radii, circumcenters, faces = compute_face_radii_and_circumcenters(points)
                if face_radii is not None and faces is not None:
                    points = combined_step(points, alpha * 0.5, face_radii, circumcenters, faces)
                    continue

            dot = np.dot(points, points.T)
            dot = np.clip(dot, -1.0, 1.0)
            np.fill_diagonal(dot, 1.0)
            dist_sq = 2.0 - 2.0 * dot
            dist_sq[dist_sq < 1e-10] = 1e-10
            w = 1.0 / (dist_sq ** 1.5)
            np.fill_diagonal(w, 0.0)
            sum_w = np.sum(w, axis=1)
            sum_p_w = w @ points
            forces = points * sum_w[:, np.newaxis] - sum_p_w
            points = points + alpha * forces
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
        return points

    def high_precision_polish(points, n_iter=100):
        """Very fine polish focusing on the largest facets with tiny steps."""
        for it in range(n_iter):
            if time.time() > deadline - 3.0:
                break
            face_radii, circumcenters, faces = compute_face_radii_and_circumcenters(points)
            if face_radii is None or faces is None:
                break
            # Use very small alpha for precision
            alpha = 0.002 * (1.0 - it / n_iter * 0.5)
            points = combined_step(points, alpha, face_radii, circumcenters, faces)
        return points

    best_points = None
    best_rad = np.inf

    n_restarts = 5

    for restart in range(n_restarts):
        if time.time() > deadline - 20.0:
            break

        if restart == 0:
            points = fibonacci_spiral(N).astype(np.float64)
        elif restart == 1:
            # Random start
            points = np.random.randn(N, 3)
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        else:
            points = best_points + 0.12 * np.random.randn(N, 3)
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms

        alpha = 0.04
        decay = 0.997
        max_iter = 400
        stall_count = 0

        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break

            # Get face radii and circumcenters for gradient-aware optimization
            face_radii, circumcenters, faces = compute_face_radii_and_circumcenters(points)

            if face_radii is not None and faces is not None:
                points = combined_step(points, alpha, face_radii, circumcenters, faces)
            else:
                points = repulsion_step_density_aware(points, alpha)

            alpha *= decay

            if it % 3 == 0:
                rad = compute_covering_radius(points)
                if rad < best_rad:
                    best_rad = rad
                    best_points = points.copy()
                    stall_count = 0
                else:
                    stall_count += 1
                    if stall_count > 20:
                        # Reduce alpha more aggressively on stall
                        alpha *= 0.95
                        stall_count = 0

            if alpha < 1e-7:
                break

        # Polish phase for this restart
        if time.time() < deadline - 15.0 and best_points is not None:
            polished = polish_step(best_points.copy(), 0.005, 40)
            rad_polished = compute_covering_radius(polished)
            if rad_polished < best_rad:
                best_rad = rad_polished
                best_points = polished

    # Final global polish with multiple stages
    if time.time() < deadline - 10.0 and best_points is not None:
        # Stage 1: moderate polish
        final = polish_step(best_points.copy(), 0.003, 60)
        rad_final = compute_covering_radius(final)
        if rad_final < best_rad:
            best_rad = rad_final
            best_points = final

        # Stage 2: high-precision polish
        if time.time() < deadline - 5.0 and best_points is not None:
            final2 = high_precision_polish(best_points.copy(), 100)
            rad_final2 = compute_covering_radius(final2)
            if rad_final2 < best_rad:
                best_rad = rad_final2
                best_points = final2

        # Stage 3: ultra-fine final polish if time permits
        if time.time() < deadline - 3.0 and best_points is not None:
            final3 = high_precision_polish(best_points.copy(), 50)
            rad_final3 = compute_covering_radius(final3)
            if rad_final3 < best_rad:
                best_rad = rad_final3
                best_points = final3

    if best_points is None:
        best_points = fibonacci_spiral(N).astype(np.float64)

    return best_points

# EVOLVE-BLOCK-END
