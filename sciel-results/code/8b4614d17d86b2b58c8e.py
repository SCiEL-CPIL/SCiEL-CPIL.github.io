import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit

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

    def _exact_objective(points_flat):
        """Exact (non-smooth) objective for optimization."""
        pts = points_flat.reshape(-1, 2)
        score = compute_score(pts)
        if score == 0.0:
            return 1e6
        return -score

    def _soft_objective(points_flat, beta):
        """
        Smooth (soft-min) objective for gradient-based optimization.
        Returns negative soft-min triangle area on unit hull area.
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

        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        if max_val < -1e6:
            return 1e6
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
        soft_min_area = -log_sum_exp / beta

        return -soft_min_area

    def _regular_polygon(n, radius=1.0, angle_offset=0.0):
        angles = np.linspace(angle_offset, 2 * np.pi + angle_offset, n, endpoint=False)
        xs = radius * np.cos(angles)
        ys = radius * np.sin(angles)
        return np.column_stack([xs, ys])

    def _refine_local(points, max_time=10.0):
        """
        Multi-stage local refinement: soft-min L-BFGS-B then coordinate descent.
        """
        pts = points.copy()
        best_score = compute_score(pts)
        best_pts = pts.copy()
        t0 = time.time()

        # Stage 1: L-BFGS-B with increasing beta
        for beta in [20.0, 100.0, 500.0]:
            if time.time() - t0 > max_time - 5:
                break
            try:
                res = opt.minimize(
                    lambda x, b=beta: _soft_objective(x, b),
                    pts.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'ftol': 1e-15, 'gtol': 1e-13}
                )
                cand = res.x.reshape(-1, 2)
                s = compute_score(cand)
                if s > best_score:
                    best_score = s
                    best_pts = cand.copy()
                    pts = cand.copy()
            except Exception:
                pass

        # Stage 2: Coordinate descent with adaptive steps
        if time.time() - t0 < max_time - 2:
            step_scales = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
            for si in range(len(step_scales)):
                if time.time() - t0 > max_time - 1:
                    break
                base_step = step_scales[si]
                improved = False
                for _pass in range(3):
                    if time.time() - t0 > max_time - 1:
                        break
                    for i in range(N):
                        best_local = pts[i].copy()
                        best_local_score = compute_score(pts)
                        directions = [
                            (base_step, 0), (-base_step, 0), (0, base_step), (0, -base_step),
                            (base_step * 0.707, base_step * 0.707), (-base_step * 0.707, base_step * 0.707),
                            (base_step * 0.707, -base_step * 0.707), (-base_step * 0.707, -base_step * 0.707),
                            (base_step * 0.3, 0), (-base_step * 0.3, 0), (0, base_step * 0.3), (0, -base_step * 0.3),
                        ]
                        for dx, dy in directions:
                            new_pt = pts.copy()
                            new_pt[i] = pts[i] + np.array([dx, dy])
                            score = compute_score(new_pt)
                            if score > best_local_score + 1e-12:
                                best_local_score = score
                                best_local = new_pt[i].copy()
                                improved = True
                        if improved:
                            pts[i] = best_local
                    if not improved:
                        break
                s = compute_score(pts)
                if s > best_score:
                    best_score = s
                    best_pts = pts.copy()

        return best_pts, best_score

    def _differential_evolution_objective(x):
        """Objective for differential evolution (minimize negative score)."""
        return _exact_objective(x)

    # Generate diverse initial points
    def _generate_initial_points(seed_offset):
        rng = np.random.RandomState(42 + seed_offset)
        points = []
        
        # Regular polygon
        points.append(_regular_polygon(N, 1.0))
        
        # Regular polygon with random rotation
        for _ in range(3):
            rot = rng.uniform(0, 2 * np.pi)
            points.append(_regular_polygon(N, 1.0, rot))
        
        # Two concentric circles
        for _ in range(3):
            r_inner = rng.uniform(0.3, 0.6)
            n_inner = rng.randint(5, 8)
            n_outer = N - n_inner
            angles_in = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + rng.uniform(0, 2 * np.pi)
            angles_out = np.linspace(0, 2 * np.pi, n_outer, endpoint=False) + rng.uniform(0, 2 * np.pi)
            pts_in = np.column_stack([r_inner * np.cos(angles_in), r_inner * np.sin(angles_in)])
            pts_out = np.column_stack([np.cos(angles_out), np.sin(angles_out)])
            p = np.vstack([pts_in, pts_out])
            points.append(_normalize_points(p))
        
        # Random points in unit disk, normalized
        for _ in range(5):
            angles = rng.uniform(0, 2 * np.pi, N)
            radii = rng.uniform(0.2, 1.0, N)
            p = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            points.append(_normalize_points(p))
        
        # Points on two arcs (symmetry breaking)
        for _ in range(2):
            half = N // 2
            remaining = N - half
            angles1 = np.linspace(0, np.pi, half, endpoint=False) + rng.uniform(-0.3, 0.3)
            angles2 = np.linspace(np.pi, 2 * np.pi, remaining, endpoint=False) + rng.uniform(-0.3, 0.3)
            radii1 = np.ones(half)
            radii2 = rng.uniform(0.8, 1.0, remaining)
            p = np.column_stack([
                np.concatenate([radii1 * np.cos(angles1), radii2 * np.cos(angles2)]),
                np.concatenate([radii1 * np.sin(angles1), radii2 * np.sin(angles2)])
            ])
            points.append(_normalize_points(p))
        
        return points

    # Phase 1: Differential Evolution for global search
    best_score = 0.0
    best_points = None

    initial_points = _generate_initial_points(0)
    
    # Use differential evolution with a reasonable population size
    pop_size = 30
    x0 = initial_points[0].flatten()
    
    # Create bounds: points should be roughly within [-2, 2]
    bounds = [(-2.0, 2.0)] * (2 * N)
    
    try:
        de_result = opt.differential_evolution(
            _differential_evolution_objective,
            bounds,
            seed=42,
            maxiter=200,
            popsize=pop_size,
            tol=1e-10,
            polish=False,
            init='sobol',
            updating='deferred'
        )
        cand = de_result.x.reshape(-1, 2)
        s = compute_score(cand)
        if s > best_score:
            best_score = s
            best_points = cand.copy()
    except Exception:
        pass

    # Phase 2: Refine the best DE result
    if best_points is not None:
        if time.time() < start_time + time_limit - 30:
            refined_pts, refined_score = _refine_local(best_points, max_time=25.0)
            if refined_score > best_score:
                best_score = refined_score
                best_points = refined_pts.copy()

    # Phase 3: Try multiple local refinements from different starting points
    if time.time() < start_time + time_limit - 20:
        for idx in range(1, min(len(initial_points), 8)):
            if time.time() > start_time + time_limit - 15:
                break
            try:
                start_pts = initial_points[idx].copy()
                refined_pts, refined_score = _refine_local(start_pts, max_time=8.0)
                if refined_score > best_score:
                    best_score = refined_score
                    best_points = refined_pts.copy()
            except Exception:
                pass

    # Phase 4: Final high-precision polish
    if best_points is not None and time.time() < start_time + time_limit - 10:
        try:
            res = opt.minimize(
                _exact_objective,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-14, 'fatol': 1e-16}
            )
            candidate = res.x.reshape(-1, 2)
            score = compute_score(candidate)
            if score > best_score:
                best_score = score
                best_points = candidate.copy()
        except Exception:
            pass

    # Phase 5: Ultra-fine coordinate descent
    if best_points is not None and time.time() < start_time + time_limit - 3:
        pts = best_points.copy()
        step_scales = [0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001]
        for base_step in step_scales:
            if time.time() > start_time + time_limit - 2:
                break
            improved = False
            for i in range(N):
                best_local = pts[i].copy()
                best_local_score = compute_score(pts)
                directions = [
                    (base_step, 0), (-base_step, 0), (0, base_step), (0, -base_step),
                    (base_step * 0.707, base_step * 0.707), (-base_step * 0.707, base_step * 0.707),
                    (base_step * 0.707, -base_step * 0.707), (-base_step * 0.707, -base_step * 0.707),
                    (base_step * 0.3, 0), (-base_step * 0.3, 0), (0, base_step * 0.3), (0, -base_step * 0.3),
                ]
                for dx, dy in directions:
                    new_pt = pts.copy()
                    new_pt[i] = pts[i] + np.array([dx, dy])
                    score = compute_score(new_pt)
                    if score > best_local_score + 1e-14:
                        best_local_score = score
                        best_local = new_pt[i].copy()
                        improved = True
                if improved:
                    pts[i] = best_local
            s = compute_score(pts)
            if s > best_score:
                best_score = s
                best_points = pts.copy()

    if best_points is None:
        best_points = _regular_polygon(N, 1.0)

    return best_points
