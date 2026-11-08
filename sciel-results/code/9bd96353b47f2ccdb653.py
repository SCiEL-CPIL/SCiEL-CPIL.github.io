import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin for 150s limit

    # Precompute triangle indices (286 triangles)
    tri_indices = list(combinations(range(N), 3))
    tri_i = np.array([t[0] for t in tri_indices], dtype=np.int32)
    tri_j = np.array([t[1] for t in tri_indices], dtype=np.int32)
    tri_k = np.array([t[2] for t in tri_indices], dtype=np.int32)

    def compute_score(points):
        """Compute min triangle area / convex hull area."""
        try:
            pts = points - points.mean(axis=0)
            hull = ConvexHull(pts)
            area_hull = hull.volume
        except Exception:
            return 0.0
        if area_hull < 1e-9:
            return 0.0

        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        min_area = np.min(areas)
        return min_area / area_hull

    def _normalize_points(points):
        """Center and scale points to unit convex hull area."""
        pts = points - points.mean(axis=0)
        try:
            hull = ConvexHull(pts)
            area = hull.volume
            if area < 1e-9:
                return pts
            pts = pts * np.sqrt(1.0 / area)
        except Exception:
            pass
        return pts

    def _soft_objective(points_flat, beta):
        """
        Smooth (soft-min) objective for gradient-based optimization.
        Returns negative soft-min triangle area on unit hull area.
        Soft min is differentiable, so L-BFGS-B can use gradients.
        """
        pts = points_flat.reshape(-1, 2)
        pts = _normalize_points(pts)

        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross

        # Log-sum-exp trick for numerical stability
        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        if max_val < -1e6:
            return 1e6
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
        soft_min_area = -log_sum_exp / beta

        return -soft_min_area

    def _exact_objective(points_flat):
        """Exact (non-smooth) objective for final polish."""
        pts = points_flat.reshape(-1, 2)
        score = compute_score(pts)
        if score == 0.0:
            return 1e6
        return -score

    def _regular_polygon(n, radius=1.0, angle_offset=0.0):
        if n == 1:
            return np.array([[0.0, 0.0]])
        angles = np.linspace(angle_offset, 2 * np.pi + angle_offset, n, endpoint=False)
        xs = radius * np.cos(angles)
        ys = radius * np.sin(angles)
        return np.column_stack([xs, ys])

    def _coordinate_descent(points, max_iter=200, tol=1e-11):
        """
        Coordinate descent: move one point at a time with adaptive step sizes.
        Exploits the structure that the min area is often determined by a few
        specific triangles, allowing targeted improvement.
        """
        pts = points.copy()
        best_score = compute_score(pts)

        # Start with larger steps, shrink as we converge
        step_scales = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]

        for iteration in range(max_iter):
            if time.time() > start_time + time_limit - 2:
                break

            # Use progressively smaller step sizes
            step_idx = min(iteration // 15, len(step_scales) - 1)
            base_step = step_scales[step_idx]

            improved = False
            for i in range(N):
                best_local = pts[i].copy()
                best_local_score = best_score

                directions = [
                    (base_step, 0), (-base_step, 0), (0, base_step), (0, -base_step),
                    (base_step * 0.707, base_step * 0.707), (-base_step * 0.707, base_step * 0.707),
                    (base_step * 0.707, -base_step * 0.707), (-base_step * 0.707, -base_step * 0.707),
                    (base_step * 0.866, base_step * 0.5), (-base_step * 0.866, base_step * 0.5),
                    (base_step * 0.866, -base_step * 0.5), (-base_step * 0.866, -base_step * 0.5),
                    # Smaller refinement steps
                    (base_step * 0.3, 0), (-base_step * 0.3, 0), (0, base_step * 0.3), (0, -base_step * 0.3),
                ]

                for dx, dy in directions:
                    new_pt = pts.copy()
                    new_pt[i] = pts[i] + np.array([dx, dy])
                    score = compute_score(new_pt)
                    if score > best_local_score + tol:
                        best_local_score = score
                        best_local = new_pt[i].copy()
                        improved = True

                if improved:
                    pts[i] = best_local
                    best_score = best_local_score

            if not improved:
                break

        return pts, best_score

    # Build diverse initial configurations (reduced to ~15 for more time per candidate)
    initializations = []

    # 1. Regular 13-gon
    initializations.append(_regular_polygon(13, 1.0, 0.0))

    # 2. 12 outer + 1 center
    initializations.append(np.vstack([_regular_polygon(12, 1.0, 0.0), np.array([[0.0, 0.0]])]))

    # 3. 11 outer + 2 inner (dipole)
    for r in [0.3, 0.5]:
        outer = _regular_polygon(11, 1.0, 0.0)
        inner = np.array([[-r, 0.0], [r, 0.0]])
        initializations.append(np.vstack([outer, inner]))

    # 4. 10 outer + 3 inner (triangle)
    for r in [0.3, 0.5]:
        outer = _regular_polygon(10, 1.0, 0.0)
        inner = _regular_polygon(3, r, np.pi / 2)
        initializations.append(np.vstack([outer, inner]))

    # 5. 9 outer + 4 inner (square)
    for r in [0.3, 0.5]:
        outer = _regular_polygon(9, 1.0, 0.0)
        inner = _regular_polygon(4, r, np.pi / 4)
        initializations.append(np.vstack([outer, inner]))

    # 6. 8 outer + 5 inner (pentagon)
    for r in [0.3, 0.5]:
        outer = _regular_polygon(8, 1.0, 0.0)
        inner = _regular_polygon(5, r, np.pi / 2)
        initializations.append(np.vstack([outer, inner]))

    # 7. 7 outer + 6 inner (hexagon)
    for r in [0.3, 0.5]:
        outer = _regular_polygon(7, 1.0, 0.0)
        inner = _regular_polygon(6, r, 0.0)
        initializations.append(np.vstack([outer, inner]))

    # 8. 6 outer + 1 center + 6 inner
    for r in [0.3, 0.5]:
        outer = _regular_polygon(6, 1.0, 0.0)
        center = np.array([[0.0, 0.0]])
        inner = _regular_polygon(6, r, np.pi / 6)
        initializations.append(np.vstack([outer, center, inner]))

    # 9. Random starts (2)
    rng = np.random.RandomState(42)
    for _ in range(2):
        rand_pts = rng.randn(N, 2) * 0.5
        initializations.append(rand_pts)

    best_score = -float('inf')
    best_points = None

    # Optimize each initialization
    for init in initializations:
        if time.time() > start_time + time_limit - 12:
            break

        current = init.copy()

        # Step 1: L-BFGS-B with annealed soft objective (differentiable)
        # Start with low beta (smooth, easy to optimize) and increase (sharper)
        betas = [10.0, 30.0, 80.0, 200.0, 500.0, 1500.0]
        for beta in betas:
            if time.time() > start_time + time_limit - 8:
                break
            try:
                res = opt.minimize(
                    lambda x, b=beta: _soft_objective(x, b),
                    current.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 80, 'ftol': 1e-16, 'gtol': 1e-14}
                )
                current = res.x.reshape(-1, 2)
            except Exception:
                pass

        # Step 2: Coordinate descent refinement
        if time.time() > start_time + time_limit - 6:
            break
        try:
            current, score_cd = _coordinate_descent(current, max_iter=300)
            if score_cd > best_score:
                best_score = score_cd
                best_points = current.copy()
        except Exception:
            pass

        # Step 3: Nelder-Mead with exact objective for final polish
        if time.time() > start_time + time_limit - 4:
            break
        try:
            res = opt.minimize(
                _exact_objective,
                current.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            candidate = res.x.reshape(-1, 2)
            score = compute_score(candidate)

            if score > best_score:
                best_score = score
                best_points = candidate.copy()
        except Exception:
            pass

        # Step 4: Final high-precision coordinate descent pass
        if time.time() > start_time + time_limit - 2:
            break
        try:
            candidate, score_cd2 = _coordinate_descent(candidate, max_iter=150)
            if score_cd2 > best_score:
                best_score = score_cd2
                best_points = candidate.copy()
        except Exception:
            pass

    # Fallback
    if best_points is None:
        best_points = initializations[0]

    return best_points
