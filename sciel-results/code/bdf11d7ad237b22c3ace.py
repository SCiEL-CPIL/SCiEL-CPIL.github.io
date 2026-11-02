import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull

N = 13

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin

    # Pre-compute all triples (i, j, k) with i < j < k
    triple_i, triple_j, triple_k = [], [], []
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                triple_i.append(i)
                triple_j.append(j)
                triple_k.append(k)
    triple_i = np.array(triple_i, dtype=np.int32)
    triple_j = np.array(triple_j, dtype=np.int32)
    triple_k = np.array(triple_k, dtype=np.int32)
    n_triples = len(triple_i)

    def compute_score_fast(points):
        """
        Compute min triangle area / convex hull area.
        points: (N, 2) array
        Returns (score, area_hull)
        """
        try:
            hull = ConvexHull(points)
            area_hull = hull.volume
        except Exception:
            return 0.0, 0.0

        if area_hull < 1e-12:
            return 0.0, 0.0

        x = points[:, 0]
        y = points[:, 1]

        xi = x[triple_i]
        xj = x[triple_j]
        xk = x[triple_k]
        yi = y[triple_i]
        yj = y[triple_j]
        yk = y[triple_k]

        cross = np.abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
        areas = 0.5 * cross
        min_area = np.min(areas)

        score = min_area / area_hull
        return score, area_hull

    def _regular_polygon(n, radius, center=(0.0, 0.0), angle_offset=0.0):
        if n <= 1:
            if n == 1:
                return np.array([[center[0], center[1]]])
            return np.array([]).reshape(0, 2)
        angles = np.linspace(angle_offset, 2 * np.pi + angle_offset, n, endpoint=False)
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)
        return np.column_stack([xs, ys])

    def make_candidate(n_hull, n_inner, hull_radius, inner_radius,
                       hull_offset=0.0, inner_offset=0.0,
                       inner_center=(0.0, 0.0), jitter=0.0, seed=None):
        if n_hull + n_inner != N:
            return None
        if n_hull < 3:
            return None
        if hull_radius <= 0:
            return None

        pts = _regular_polygon(n_hull, hull_radius, angle_offset=hull_offset)
        if n_inner > 0:
            inner = _regular_polygon(n_inner, inner_radius,
                                     center=inner_center, angle_offset=inner_offset)
            pts = np.vstack([pts, inner])

        if pts.shape[0] != N:
            return None

        if jitter > 0:
            rng = np.random.default_rng(seed)
            pts = pts + rng.normal(0, jitter, pts.shape)

        return pts

    # ------------------------------------------------------------------
    # Phase 1: Generate and evaluate structured candidates
    # ------------------------------------------------------------------
    best_points = None
    best_score = -float('inf')
    candidates = []

    # Systematic sweep
    hull_radii = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    inner_radii = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    offsets = [0.0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2]

    for n_hull in range(3, N + 1):
        n_inner = N - n_hull
        if n_inner < 0:
            continue

        for hr in hull_radii:
            for ir in inner_radii:
                if n_inner == 0:
                    for ho in offsets:
                        if time.time() - start_time > time_limit - 30:
                            break
                        cand = make_candidate(n_hull, 0, hr, 0.0, hull_offset=ho)
                        if cand is None:
                            continue
                        candidates.append(cand)
                    if time.time() - start_time > time_limit - 30:
                        break
                else:
                    for ho in offsets:
                        for io in offsets:
                            if time.time() - start_time > time_limit - 30:
                                break
                            cand = make_candidate(n_hull, n_inner, hr, ir,
                                                  hull_offset=ho, inner_offset=io)
                            if cand is None:
                                continue
                            candidates.append(cand)
                        if time.time() - start_time > time_limit - 30:
                            break
                    if time.time() - start_time > time_limit - 30:
                        break

        if time.time() - start_time > time_limit - 30:
            break

    # Add jittered variants
    if time.time() - start_time < time_limit - 30:
        rng = np.random.default_rng(42)
        for n_hull in range(6, N + 1):
            n_inner = N - n_hull
            if n_inner < 0:
                continue
            for trial in range(5):
                hr = rng.uniform(0.8, 1.2)
                ir = rng.uniform(0.1, 0.5) if n_inner > 0 else 0.0
                ho = rng.uniform(0, 2 * np.pi)
                io = rng.uniform(0, 2 * np.pi)
                jitter = rng.uniform(0.02, 0.08)
                cand = make_candidate(n_hull, n_inner, hr, ir,
                                      hull_offset=ho, inner_offset=io,
                                      jitter=jitter, seed=trial * 100 + n_hull)
                if cand is not None:
                    candidates.append(cand)
            if time.time() - start_time > time_limit - 30:
                break

    # Evaluate candidates
    for idx, cand in enumerate(candidates):
        if time.time() - start_time > time_limit - 25:
            break
        s, _ = compute_score_fast(cand)
        if s > best_score:
            best_score = s
            best_points = cand.copy()

    # ------------------------------------------------------------------
    # Phase 2: Local refinement using Nelder-Mead
    # ------------------------------------------------------------------
    if best_points is not None and time.time() - start_time < time_limit - 10:
        def hard_obj(x):
            pts = x.reshape(-1, 2)
            s, _ = compute_score_fast(pts)
            return -s

        try:
            res = opt.minimize(
                hard_obj,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'adaptive': True,
                         'xatol': 1e-12, 'fatol': 1e-14}
            )
            nm_pts = res.x.reshape(-1, 2)
            nm_score, _ = compute_score_fast(nm_pts)
            if nm_score > best_score:
                best_score = nm_score
                best_points = nm_pts.copy()
        except Exception:
            pass

        # L-BFGS-B with smooth surrogate
        if time.time() - start_time < time_limit - 5:
            beta = 100.0

            def soft_obj(x):
                pts = x.reshape(-1, 2)
                try:
                    hull = ConvexHull(pts)
                    area_hull = hull.volume
                except Exception:
                    return 1e6
                if area_hull < 1e-12:
                    return 1e6

                xi = pts[triple_i, 0]
                xj = pts[triple_j, 0]
                xk = pts[triple_k, 0]
                yi = pts[triple_i, 1]
                yj = pts[triple_j, 1]
                yk = pts[triple_k, 1]

                cross = np.abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
                areas = 0.5 * cross

                neg_beta = -beta * areas
                max_val = np.max(neg_beta)
                lse = max_val + np.log(np.sum(np.exp(neg_beta - max_val)))
                soft_min = -lse / beta
                return -(soft_min / area_hull)

            try:
                res = opt.minimize(
                    soft_obj,
                    best_points.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                lb_pts = res.x.reshape(-1, 2)
                lb_score, _ = compute_score_fast(lb_pts)
                if lb_score > best_score:
                    best_score = lb_score
                    best_points = lb_pts.copy()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Phase 3: Final polish
    # ------------------------------------------------------------------
    if best_points is not None and time.time() - start_time < time_limit - 2:
        try:
            res = opt.minimize(
                hard_obj,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 300, 'adaptive': True,
                         'xatol': 1e-13, 'fatol': 1e-15}
            )
            final_pts = res.x.reshape(-1, 2)
            final_score, _ = compute_score_fast(final_pts)
            if final_score > best_score:
                best_score = final_score
                best_points = final_pts.copy()
        except Exception:
            pass

    if best_points is None:
        best_points = _regular_polygon(13, radius=1.0)

    return best_points
