"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for safety

    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
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

    def compute_face_radii(pts):
        """Return array of face circumradii (in radians) for gradient-aware optimization."""
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

            return radii, faces
        except Exception:
            return None, None

    def repulsion_step_density_aware(points, alpha, face_radii=None, faces=None):
        """Repulsion with density-aware weighting: points in large-cap regions get stronger push."""
        dot = np.dot(points, points.T)
        dot = np.clip(dot, -1.0, 1.0)
        np.fill_diagonal(dot, 1.0)
        dist_sq = 2.0 - 2.0 * dot
        dist_sq[dist_sq < 1e-10] = 1e-10
        w = 1.0 / (dist_sq ** 1.5)
        np.fill_diagonal(w, 0.0)

        # Density-aware weighting: weight vertices in large faces more
        if face_radii is not None and faces is not None:
            vertex_max_rad = np.zeros(len(points))
            for fi in range(len(faces)):
                for vi in faces[fi]:
                    if face_radii[fi] > vertex_max_rad[vi]:
                        vertex_max_rad[vi] = face_radii[fi]
            if np.max(vertex_max_rad) > 1e-10:
                weight_boost = 1.0 + 0.5 * vertex_max_rad / np.max(vertex_max_rad)
                w = w * weight_boost[:, np.newaxis]

        sum_w = np.sum(w, axis=1)
        sum_p_w = w @ points
        forces = points * sum_w[:, np.newaxis] - sum_p_w
        points = points + alpha * forces
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        return points

    def targeted_face_reduction(points, n_iter=200):
        """Directly reduce the largest facet radii by pushing vertices of large facets apart."""
        best = points.copy()
        best_rad = compute_covering_radius(best)

        for it in range(n_iter):
            if time.time() > deadline - 5.0:
                break

            face_radii, faces = compute_face_radii(points)
            if face_radii is None:
                break

            # Find the largest facets and push their vertices apart
            top_k = min(15, len(faces))
            top_indices = np.argsort(face_radii)[-top_k:]

            # For each large facet, compute the circumcenter direction and push vertices away
            displacement = np.zeros_like(points)
            for fi in top_indices:
                if face_radii[fi] < best_rad * 0.85:
                    continue
                f = faces[fi]
                a, b, c = points[f[0]], points[f[1]], points[f[2]]
                # Compute circumcenter of the triangle on the sphere
                ab = np.cross(a, b)
                bc = np.cross(b, c)
                ca = np.cross(c, a)
                c_dir = ab + bc + ca
                norm_c = np.linalg.norm(c_dir)
                if norm_c < 1e-10:
                    continue
                c_dir = c_dir / norm_c

                # Push each vertex away from the circumcenter (toward reducing the cap)
                for vi in f:
                    p = points[vi]
                    # Move away from circumcenter in tangent space
                    away = p - np.dot(p, c_dir) * c_dir
                    away_norm = np.linalg.norm(away)
                    if away_norm > 1e-10:
                        away /= away_norm
                        displacement[vi] += 0.01 * away

            if np.linalg.norm(displacement) < 1e-10:
                continue

            points = points + displacement
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms

            new_rad = compute_covering_radius(points)
            if new_rad < best_rad:
                best = points.copy()
                best_rad = new_rad

        return best, best_rad

    def local_search(points, max_time=30.0):
        """Local search: try small perturbations of individual points."""
        start_time = time.time()
        points = np.array(points, dtype=np.float64)
        n = len(points)
        best_radius = compute_covering_radius(points)
        best_points = points.copy()

        rng = np.random.RandomState(12345)

        while time.time() - start_time < max_time and time.time() < deadline - 5.0:
            # Pick a random point to perturb
            idx = rng.randint(n)

            # Generate a random perturbation direction
            perturbation = rng.randn(3)
            # Project to tangent space
            tangent = perturbation - np.dot(perturbation, points[idx]) * points[idx]
            tnorm = np.linalg.norm(tangent)
            if tnorm < 1e-10:
                continue
            tangent /= tnorm

            # Try several step sizes
            improved = False
            for step_size in [0.05, 0.02, 0.01, 0.005, 0.002]:
                new_points = points.copy()
                new_points[idx] = points[idx] + step_size * tangent
                new_norm = np.linalg.norm(new_points[idx])
                new_points[idx] /= new_norm

                new_radius = compute_covering_radius(new_points)
                if new_radius < best_radius - 1e-6:
                    points = new_points
                    best_points = points.copy()
                    best_radius = new_radius
                    improved = True
                    break

            if not improved:
                # Try the opposite direction
                for step_size in [0.05, 0.02, 0.01, 0.005, 0.002]:
                    new_points = points.copy()
                    new_points[idx] = points[idx] - step_size * tangent
                    new_norm = np.linalg.norm(new_points[idx])
                    new_points[idx] /= new_norm

                    new_radius = compute_covering_radius(new_points)
                    if new_radius < best_radius - 1e-6:
                        points = new_points
                        best_points = points.copy()
                        best_radius = new_radius
                        improved = True
                        break

        return best_points, best_radius

    def polish(points, n_iter=50):
        """High-precision polish with very small steps."""
        for _ in range(n_iter):
            if time.time() > deadline - 5.0:
                break
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
            points = points + 0.001 * forces
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            points = points / norms
        return points

    # Start with Fibonacci spiral
    points = fibonacci_spiral(N)

    # Phase 1: Density-aware repulsion to spread points
    face_radii, faces = compute_face_radii(points)
    for i in range(100):
        if time.time() > deadline - 80.0:
            break
        alpha = 0.01 * (0.95 ** (i // 10))
        points = repulsion_step_density_aware(points, alpha, face_radii, faces)
        if i % 10 == 0:
            face_radii, faces = compute_face_radii(points)

    # Phase 2: Targeted face reduction
    points, radius = targeted_face_reduction(points, n_iter=150)

    # Check time
    if time.time() > deadline - 40.0:
        # Not enough time for local search
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        return points / norms

    # Phase 3: Local search
    points, radius = local_search(points, max_time=25.0)

    # Phase 4: Final polish
    if time.time() < deadline - 10.0:
        points = polish(points, n_iter=30)

    # Final normalization
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms

    # Ensure no coincident points
    for i in range(N):
        for j in range(i + 1, N):
            if np.linalg.norm(points[i] - points[j]) < 1e-8:
                # Nudge apart
                rng = np.random.RandomState(42 + i)
                points[j] += 1e-6 * rng.randn(3)
                points[j] /= np.linalg.norm(points[j])

    return points

# EVOLVE-BLOCK-END
