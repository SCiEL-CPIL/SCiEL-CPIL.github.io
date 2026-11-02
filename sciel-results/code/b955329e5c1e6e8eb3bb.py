"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull

def construct():
    start_time = time.time()
    time_limit = 140.0  # Leave 10s margin for safety before 150s kill

    # Precompute triple index arrays for vectorized area computation
    _tri_i = []
    _tri_j = []
    _tri_k = []
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                _tri_i.append(i)
                _tri_j.append(j)
                _tri_k.append(k)
    tri_i = np.array(_tri_i, dtype=np.int32)
    tri_j = np.array(_tri_j, dtype=np.int32)
    tri_k = np.array(_tri_k, dtype=np.int32)
    n_triples = len(tri_i)

    def compute_score(points):
        """Compute min triangle area / convex hull area for given points."""
        try:
            hull = ConvexHull(points)
            area_hull = hull.volume
        except Exception:
            return 0.0, 0.0

        if area_hull < 1e-9:
            return 0.0, 0.0

        x = points[:, 0]
        y = points[:, 1]
        # Vectorized cross products for all 286 triples
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
        min_area = float(np.min(areas))

        score = min_area / area_hull
        return score, area_hull

    def _soft_min_score(points, beta=50.0):
        """
        Smooth surrogate: use log-sum-exp to approximate the minimum triangle area.
        Returns negative soft-min score for minimization.
        """
        try:
            hull = ConvexHull(points)
            area_hull = hull.volume
        except Exception:
            return 1e6

        if area_hull < 1e-9:
            return 1e6

        x = points[:, 0]
        y = points[:, 1]

        # Vectorized triangle areas
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)

        # Soft-min via log-sum-exp (stable)
        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        if max_val < -1e6:
            return 1e6
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
        soft_min_area = -log_sum_exp / beta

        score = soft_min_area / area_hull
        return -score

    def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
        """Generate points of a regular n-gon."""
        if n == 1:
            return np.array([center])
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)
        return np.column_stack([xs, ys])

    def _ring_config(r1, r2, r3, n1, n2, n3, phase1=0.0, phase2=0.0, phase3=0.0):
        """
        Symmetric ring configuration with 3 rings.
        n1 points on ring 1 (radius r1), n2 on ring 2 (radius r2), n3 on ring 3 (radius r3).
        If n3==1, it's a single center point at (0,0).
        """
        pts = []
        if n1 > 0:
            a = np.linspace(phase1, phase1 + 2 * np.pi, n1, endpoint=False)
            pts.append(np.column_stack([r1 * np.cos(a), r1 * np.sin(a)]))
        if n2 > 0:
            a = np.linspace(phase2, phase2 + 2 * np.pi, n2, endpoint=False)
            pts.append(np.column_stack([r2 * np.cos(a), r2 * np.sin(a)]))
        if n3 == 1:
            pts.append(np.array([[0.0, 0.0]]))
        elif n3 > 1:
            a = np.linspace(phase3, phase3 + 2 * np.pi, n3, endpoint=False)
            pts.append(np.column_stack([r3 * np.cos(a), r3 * np.sin(a)]))
        if not pts:
            return _regular_polygon(N, radius=1.0)
        return np.vstack(pts)

    def _optimize_from(start_pts, time_budget):
        """Run multi-stage optimization from a starting configuration."""
        best = start_pts.copy()
        best_s, _ = compute_score(best)
        current = start_pts.copy()

        # Stage 1: L-BFGS-B with annealed soft-min beta
        for beta in [15.0, 40.0, 80.0, 150.0]:
            if time.time() - start_time > time_budget:
                break
            try:
                res = opt.minimize(
                    lambda x: _soft_min_score(x.reshape(-1, 2), beta),
                    current.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 60, 'ftol': 1e-13, 'gtol': 1e-11}
                )
                cand = res.x.reshape(-1, 2)
                s, _ = compute_score(cand)
                if s > best_s:
                    best_s = s
                    best = cand.copy()
                current = cand
            except Exception:
                pass

        # Stage 2: Nelder-Mead polish on hard objective
        if time.time() - start_time < time_budget - 3:
            try:
                res = opt.minimize(
                    lambda x: -compute_score(x.reshape(-1, 2))[0],
                    best.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'adaptive': True, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                cand = res.x.reshape(-1, 2)
                s, _ = compute_score(cand)
                if s > best_s:
                    best_s = s
                    best = cand.copy()
            except Exception:
                pass

        # Stage 3: Fine-tune with Powell on hard objective
        if time.time() - start_time < time_budget - 1:
            try:
                res = opt.minimize(
                    lambda x: -compute_score(x.reshape(-1, 2))[0],
                    best.flatten(),
                    method='Powell',
                    options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-14}
                )
                cand = res.x.reshape(-1, 2)
                s, _ = compute_score(cand)
                if s > best_s:
                    best_s = s
                    best = cand.copy()
            except Exception:
                pass

        return best, best_s

    best_points = None
    best_score = -float('inf')

    # --- Phase 1: Symmetric ring initializations (low-dimensional, smooth landscape) ---
    # These capture the geometry that the cell findings indicate is near-optimal.
    ring_starts = [
        # 1 center + 12 outer
        _ring_config(r1=1.0, r2=0.0, r3=0.0, n1=12, n2=0, n3=1),
        # 1 center + 6 mid + 6 outer
        _ring_config(r1=1.0, r2=0.5, r3=0.0, n1=6, n2=6, n3=1),
        # 1 center + 5 mid + 7 outer (asymmetric split)
        _ring_config(r1=1.0, r2=0.45, r3=0.0, n1=7, n2=5, n3=1),
        # 1 center + 4 mid + 8 outer
        _ring_config(r1=1.0, r2=0.4, r3=0.0, n1=8, n2=4, n3=1),
        # 1 center + 12 outer with slight phase offset on outer
        _ring_config(r1=1.0, r2=0.0, r3=0.0, n1=12, n2=0, n3=1, phase1=np.pi/12),
        # 2 center + 11 outer (approx)
        _ring_config(r1=1.0, r2=0.3, r3=0.0, n1=11, n2=2, n3=0, phase2=np.pi/2),
        # 3 center + 10 outer
        _ring_config(r1=1.0, r2=0.4, r3=0.0, n1=10, n2=3, n3=0, phase2=np.pi/3),
        # 1 center + 3 mid + 9 outer
        _ring_config(r1=1.0, r2=0.6, r3=0.0, n1=9, n2=3, n3=1),
        # 1 center + 2 mid + 10 outer
        _ring_config(r1=1.0, r2=0.7, r3=0.0, n1=10, n2=2, n3=1),
        # 1 center + 4 mid + 8 outer with different phase
        _ring_config(r1=1.0, r2=0.5, r3=0.0, n1=8, n2=4, n3=1, phase2=np.pi/4),
        # 2 concentric rings without center: 7 + 6
        _ring_config(r1=1.0, r2=0.55, r3=0.0, n1=7, n2=6, n3=0),
        # 2 concentric rings without center: 8 + 5
        _ring_config(r1=1.0, r2=0.45, r3=0.0, n1=8, n2=5, n3=0),
        # 3 rings: 5 + 4 + 4
        _ring_config(r1=1.0, r2=0.65, r3=0.35, n1=5, n2=4, n3=4),
        # 3 rings: 6 + 4 + 3
        _ring_config(r1=1.0, r2=0.6, r3=0.3, n1=6, n2=4, n3=3),
        # 1 center + 12 outer, rotated
        _ring_config(r1=1.0, r2=0.0, r3=0.0, n1=12, n2=0, n3=1, phase1=np.pi/6),
    ]

    # Also add some random starts for diversity
    np.random.seed(42)
    for _ in range(5):
        ring_starts.append(np.random.rand(N, 2))

    # Optimize from each starting configuration
    for i, start_pts in enumerate(ring_starts):
        if time.time() - start_time > time_limit - 10:
            break

        # Allocate time budget based on remaining time and number of starts left
        remaining_starts = len(ring_starts) - i
        remaining_time = time_limit - (time.time() - start_time)
        time_per_start = min(remaining_time / max(remaining_starts, 1), 20.0)

        try:
            pts, score = _optimize_from(start_pts, time.time() - start_time + time_per_start)
            if score > best_score:
                best_score = score
                best_points = pts.copy()
        except Exception:
            pass

    # Final global refinement if we have time
    if best_points is not None and time.time() - start_time < time_limit - 5:
        try:
            final_pts, final_score = _optimize_from(best_points, time_limit)
            if final_score > best_score:
                best_score = final_score
                best_points = final_pts.copy()
        except Exception:
            pass

    # Return best result found
    if best_points is not None:
        return best_points
    else:
        # Fallback: return a regular 13-gon
        return _regular_polygon(N, radius=1.0)

# EVOLVE-BLOCK-END
