import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull

N = 13

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin (evaluator kills at 150s)

    def get_triples_indices():
        """Return index arrays for all triples (vectorized)."""
        i, j, k = np.meshgrid(
            np.arange(N), np.arange(N), np.arange(N), indexing='ij'
        )
        mask = (i < j) & (j < k)
        return i[mask], j[mask], k[mask]

    triple_i, triple_j, triple_k = get_triples_indices()
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
        """
        Build a candidate configuration.
        n_hull: number of points on outer ring
        n_inner: number of interior points (0 allowed)
        Returns (N, 2) array or None if invalid
        """
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
        else:
            pts = pts

        if pts.shape[0] != N:
            return None

        if jitter > 0:
            rng = np.random.default_rng(seed)
            pts = pts + rng.normal(0, jitter, pts.shape)

        return pts

    def soft_min_objective(x, beta=200.0):
        """
        Smooth surrogate: -soft_min(areas) / hull_area
        Uses log-sum-exp for numerical stability.
        """
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

        # Soft min via log-sum-exp
        neg_beta = -beta * areas
        max_val = np.max(neg_beta)
        lse = max_val + np.log(np.sum(np.exp(neg_beta - max_val)))
        soft_min = -lse / beta
        return -(soft_min / area_hull)

    # ------------------------------------------------------------------
    # Phase 1: Focused grid search on high-symmetry configurations
    # ------------------------------------------------------------------
    best_points = None
    best_score = -float('inf')
    candidates = []

    # Focus on n_hull values that are divisors or near-divisors of N for symmetry
    # N=13 is prime, so we look at n_hull = 3,4,5,6,7,8,9,10,11,12,13
    # But most promising: regular 13-gon, or 6+7, 7+6, 5+8, 8+5 splits
    priority_hulls = [13, 7, 6, 8, 5, 9, 4, 10, 11, 12, 3]

    hull_radii = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    inner_radii = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
    offsets = [0.0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2, np.pi / 12]

    for n_hull in priority_hulls:
        n_inner = N - n_hull
        if n_inner < 0:
            continue

        for hr in hull_radii:
            for ir in inner_radii:
                if n_inner == 0:
                    for ho in offsets:
                        if time.time() - start_time > time_limit - 40:
                            break
                        cand = make_candidate(n_hull, 0, hr, 0.0,
                                              hull_offset=ho)
                        if cand is not None:
                            candidates.append(cand)
                    if time.time() - start_time > time_limit - 40:
                        break
                else:
                    for ho in offsets:
                        for io in offsets:
                            if time.time() - start_time > time_limit - 40:
                                break
                            cand = make_candidate(n_hull, n_inner, hr, ir,
                                                  hull_offset=ho, inner_offset=io)
                            if cand is not None:
                                candidates.append(cand)
                        if time.time() - start_time > time_limit - 40:
                            break
                    if time.time() - start_time > time_limit - 40:
                        break

        if time.time() - start_time > time_limit - 40:
            break

    # Add jittered variants for top structures
    if time.time() - start_time < time_limit - 35:
        rng = np.random.default_rng(42)
        for n_hull in [13, 7, 6, 8, 5]:
            n_inner = N - n_hull
            if n_inner < 0:
                continue
            for trial in range(8):
                hr = rng.uniform(0.8, 1.2)
                ir = rng.uniform(0.1, 0.5) if n_inner > 0 else 0.0
                ho = rng.uniform(0, 2 * np.pi)
                io = rng.uniform(0, 2 * np.pi)
                jitter = rng.uniform(0.01, 0.05)
                cand = make_candidate(n_hull, n_inner, hr, ir,
                                      hull_offset=ho, inner_offset=io,
                                      jitter=jitter, seed=trial * 100 + n_hull)
                if cand is not None:
                    candidates.append(cand)
            if time.time() - start_time > time_limit - 35:
                break

    # Evaluate all candidates
    print(f"Evaluating {len(candidates)} candidates...", flush=True)
    for idx, cand in enumerate(candidates):
        if time.time() - start_time > time_limit - 30:
            break
        s, _ = compute_score_fast(cand)
        if s > best_score:
            best_score = s
            best_points = cand.copy()
        if idx % 200 == 0:
            print(f"  Candidate {idx}/{len(candidates)}: best={best_score:.6f}",
                  flush=True)

    print(f"Phase 1 done. Best score: {best_score:.6f}", flush=True)

    # ------------------------------------------------------------------
    # Phase 2: L-BFGS-B refinement with smooth surrogate (primary optimizer)
    # ------------------------------------------------------------------
    if best_points is not None and time.time() - start_time < time_limit - 15:
        # Fix P0 to remove translation invariance: pin first point at origin
        # Actually, let's work in centered coordinates and pin scale
        # Normalize: center at centroid, scale so hull area = 1
        def normalize_points(pts):
            """Center at centroid and scale to unit hull area."""
            pts = pts.copy()
            # Center
            centroid = np.mean(pts, axis=0)
            pts -= centroid
            # Scale to unit hull area
            try:
                hull = ConvexHull(pts)
                area = hull.volume
                if area > 1e-12:
                    pts *= np.sqrt(1.0 / area)
            except Exception:
                pass
            return pts

        # Work with normalized coordinates, pin P0 = (0,0) by subtracting
        # Actually simpler: just optimize directly with L-BFGS-B on soft objective
        # The soft objective is smooth and L-BFGS-B should handle it well

        # Try multiple beta values for the soft-min
        for beta in [100.0, 200.0, 400.0]:
            if time.time() - start_time > time_limit - 10:
                break

            def soft_obj_beta(x):
                return soft_min_objective(x, beta=beta)

            try:
                res = opt.minimize(
                    soft_obj_beta,
                    best_points.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12,
                             'maxls': 50}
                )
                lb_pts = res.x.reshape(-1, 2)
                lb_score, _ = compute_score_fast(lb_pts)
                if lb_score > best_score:
                    best_score = lb_score
                    best_points = lb_pts.copy()
                print(f"After L-BFGS-B (beta={beta:.0f}): {best_score:.6f}", flush=True)
            except Exception as e:
                print(f"L-BFGS-B (beta={beta:.0f}) failed: {e}", flush=True)

    # ------------------------------------------------------------------
    # Phase 3: Nelder-Mead polish on the hard objective
    # ------------------------------------------------------------------
    if best_points is not None and time.time() - start_time < time_limit - 5:
        def hard_obj(x):
            pts = x.reshape(-1, 2)
            s, _ = compute_score_fast(pts)
            return -s

        try:
            res = opt.minimize(
                hard_obj,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 300, 'adaptive': True,
                         'xatol': 1e-12, 'fatol': 1e-14}
            )
            nm_pts = res.x.reshape(-1, 2)
            nm_score, _ = compute_score_fast(nm_pts)
            if nm_score > best_score:
                best_score = nm_score
                best_points = nm_pts.copy()
            print(f"After NM polish: {best_score:.6f}", flush=True)
        except Exception as e:
            print(f"NM polish failed: {e}", flush=True)

    # ------------------------------------------------------------------
    # Phase 4: High-precision final polish
    # ------------------------------------------------------------------
    if best_points is not None and time.time() - start_time < time_limit - 2:
        def hard_obj_final(x):
            pts = x.reshape(-1, 2)
            s, _ = compute_score_fast(pts)
            return -s

        try:
            res = opt.minimize(
                hard_obj_final,
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 200, 'adaptive': True,
                         'xatol': 1e-14, 'fatol': 1e-16}
            )
            final_pts = res.x.reshape(-1, 2)
            final_score, _ = compute_score_fast(final_pts)
            if final_score > best_score:
                best_score = final_score
                best_points = final_pts.copy()
            print(f"Final high-precision polish: {best_score:.6f}", flush=True)
        except Exception as e:
            print(f"Final polish failed: {e}", flush=True)

    if best_points is None:
        best_points = _regular_polygon(13, radius=1.0)

    print(f"FINAL SCORE: {best_score:.8f}", flush=True)
    return best_points

# EVOLVE-BLOCK-END
