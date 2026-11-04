"""Heilbronn convex regions: Maximize minimum triangle area / hull area."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13
NUM_TRIPLES = N * (N - 1) * (N - 2) // 6
TIME_LIMIT = 145.0  # safety margin from 150s kill

# Precompute triple indices
_triple_indices = np.array(list(combinations(range(N), 3)))


def _all_triangle_areas(pts):
    """Vectorized: compute all C(N,3) triangle areas. Returns (areas, idx_i, idx_j, idx_k)."""
    n = len(pts)
    i = _triple_indices[:, 0]
    j = _triple_indices[:, 1]
    k = _triple_indices[:, 2]
    p1, p2, p3 = pts[i], pts[j], pts[k]
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - \
            (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    return areas, i, j, k


def _hull_area(pts):
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _score(pts):
    pts = np.asarray(pts, dtype=float)
    if pts.shape != (N, 2):
        return -1e9
    areas, _, _, _ = _all_triangle_areas(pts)
    min_area = areas.min()
    hull_a = _hull_area(pts)
    if hull_a < 1e-12:
        return -1e9
    return min_area / hull_a


def _soft_min(ratios, tau):
    """Smooth approximation of the minimum using log-sum-exp."""
    min_r = np.min(ratios)
    shifted = -tau * (ratios - min_r)
    shifted = np.clip(shifted, -50, 0)
    max_shifted = np.max(shifted)
    log_sum = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    return min_r - (1.0 / tau) * log_sum


def _soft_score(pts, tau=200.0):
    """Smooth approximation of _score for gradient-based optimization."""
    pts = np.asarray(pts, dtype=float)
    if pts.shape != (N, 2):
        return 0.0
    areas, _, _, _ = _all_triangle_areas(pts)
    hull_a = _hull_area(pts)
    if hull_a < 1e-12:
        return 0.0
    ratios = areas / hull_a
    return _soft_min(ratios, tau)


# ---------- Edge-vector convex polygon parameterization ----------

def _edges_to_points(log_lengths, angle_devs):
    """
    log_lengths: (N,) log of edge lengths
    angle_devs: (N-1,) deviations of turning angles from 2*pi/N (last is determined)
    Returns (N,2) polygon vertices (closed, starting at origin).
    """
    base_angle = 2.0 * np.pi / N
    full_devs = np.zeros(N)
    full_devs[:-1] = angle_devs
    full_devs[-1] = -np.sum(angle_devs)
    angles = base_angle + full_devs
    angles = np.maximum(angles, 1e-6)
    angles = angles * (2.0 * np.pi / angles.sum())

    lengths = np.exp(log_lengths)
    cum = np.cumsum(np.concatenate([[0.0], angles[:-1]]))
    dx = lengths * np.cos(cum)
    dy = lengths * np.sin(cum)
    verts = np.zeros((N + 1, 2))
    verts[1:, 0] = np.cumsum(dx)
    verts[1:, 1] = np.cumsum(dy)
    pts = verts[:N].copy()
    pts -= pts.mean(axis=0)
    return pts


def _points_to_edges(pts):
    """
    Convert a convex polygon (given in CCW order) to (log_lengths, angle_devs).
    Returns None if not convex / not closed.
    """
    pts = np.asarray(pts, dtype=float)
    if pts.shape != (N, 2):
        return None
    edges = np.roll(pts, -1, axis=0) - pts
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(lengths < 1e-12):
        return None
    angles = np.zeros(N)
    for i in range(N):
        e1 = edges[i]
        e2 = edges[(i + 1) % N]
        cross = e1[0] * e2[1] - e1[1] * e2[0]
        dot = e1[0] * e2[0] + e1[1] * e2[1]
        ang = np.arctan2(cross, dot)
        if ang < 0:
            ang += 2.0 * np.pi
        angles[i] = ang
    if np.any(angles <= 1e-6) or np.any(angles >= np.pi - 1e-6):
        return None
    if abs(angles.sum() - 2.0 * np.pi) > 1e-3:
        return None
    base_angle = 2.0 * np.pi / N
    devs = angles - base_angle
    log_lengths = np.log(lengths)
    angle_devs = devs[:-1].copy()
    return log_lengths, angle_devs


def _regular_polygon_params():
    """Regular 13-gon parameters."""
    log_lengths = np.zeros(N)
    angle_devs = np.zeros(N - 1)
    return log_lengths, angle_devs


def _param_to_score(log_lengths, angle_devs, tau=None):
    pts = _edges_to_points(log_lengths, angle_devs)
    if tau is None:
        return _score(pts)
    return _soft_score(pts, tau=tau)


def _optimize_from_params(log_lengths, angle_devs, start_time, time_budget, tau=200.0):
    """
    Run L-BFGS-B on the edge parameters to maximize soft score.
    Returns (best_pts, best_score).
    """
    def neg_obj(x):
        ll = x[:N]
        ad = x[N:2 * N - 1]
        return -_param_to_score(ll, ad, tau=tau)

    x0 = np.concatenate([log_lengths, angle_devs])
    bounds = [(-3.0, 3.0)] * N + [(-np.pi / 2, np.pi / 2)] * (N - 1)

    best_pts = _edges_to_points(log_lengths, angle_devs)
    best_score = _score(best_pts)

    t0 = time.time()
    while (time.time() - t0) < time_budget:
        try:
            res = minimize(
                neg_obj, x0, method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 3000, 'maxfun': 100000, 'disp': False}
            )
            x0 = res.x
            ll = x0[:N]
            ad = x0[N:2 * N - 1]
            pts = _edges_to_points(ll, ad)
            sc = _score(pts)
            if sc > best_score:
                best_score = sc
                best_pts = pts.copy()
        except Exception:
            break
        if res.success and res.status == 0:
            break

    return best_pts, best_score


def _coordinate_descent_min_triangle(pts, start_time, time_budget, max_rounds=2000):
    """
    Local refinement: repeatedly find the minimum-area triangle and
    nudge one of its vertices to increase that triangle's area.
    """
    pts = np.array(pts, dtype=float)
    best_score = _score(pts)
    best_pts = pts.copy()
    t0 = time.time()

    for _ in range(max_rounds):
        if (time.time() - t0) > time_budget:
            break
        areas, ii, jj, kk = _all_triangle_areas(pts)
        min_idx = np.argmin(areas)
        min_area = areas[min_idx]
        i, j, k = ii[min_idx], jj[min_idx], kk[min_idx]

        p_i, p_j, p_k = pts[i], pts[j], pts[k]
        centroid = (p_i + p_j + p_k) / 3.0

        improved = False
        for v in [i, j, k]:
            p_v = pts[v]
            direction = p_v - centroid
            norm = np.linalg.norm(direction)
            if norm < 1e-12:
                continue
            direction = direction / norm

            best_step = 0.0
            best_local = min_area
            for step in [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]:
                new_pts = pts.copy()
                new_pts[v] = p_v + step * direction
                a_new = 0.5 * abs(
                    (new_pts[j][0] - new_pts[i][0]) * (new_pts[k][1] - new_pts[i][1]) -
                    (new_pts[j][1] - new_pts[i][1]) * (new_pts[k][0] - new_pts[i][0])
                )
                if a_new > best_local:
                    best_local = a_new
                    best_step = step

            if best_step > 0:
                new_pts = pts.copy()
                new_pts[v] = p_v + best_step * direction
                sc = _score(new_pts)
                if sc > best_score:
                    best_score = sc
                    best_pts = new_pts.copy()
                    pts = new_pts.copy()
                    improved = True
                    break

        if not improved:
            rng = np.random.default_rng()
            new_pts = pts + rng.normal(0, 1e-4, size=pts.shape)
            sc = _score(new_pts)
            if sc > best_score:
                best_score = sc
                best_pts = new_pts.copy()
                pts = new_pts.copy()
            else:
                break

    return best_pts, best_pts


def _generate_starts(rng):
    """Generate diverse starting parameters for multi-start."""
    starts = []
    starts.append(_regular_polygon_params())
    for _ in range(7):
        ll = np.zeros(N) + rng.normal(0, 0.1)
        ad = rng.normal(0, 0.1, size=N - 1)
        starts.append((ll, ad))
    for _ in range(6):
        ll = np.zeros(N) + rng.normal(0, 0.3)
        ad = rng.normal(0, 0.3, size=N - 1)
        starts.append((ll, ad))
    for amp in [0.2, 0.4, 0.6]:
        ad = np.zeros(N - 1)
        for i in range(N - 1):
            ad[i] = amp * (1 if i % 2 == 0 else -1)
        ll = np.zeros(N)
        starts.append((ll, ad))
    return starts


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)

    best_pts = None
    best_score = -1e9

    def _update(pts, sc):
        nonlocal best_pts, best_score
        if pts is not None and sc > best_score:
            best_score = sc
            best_pts = np.array(pts, dtype=float)

    # Phase 1: Multi-start local refinement with annealed tau
    starts = _generate_starts(rng)
    tau_schedule = [50, 100, 200, 400, 800]

    for tau in tau_schedule:
        if (time.time() - start_time) > TIME_LIMIT - 15.0:
            break

        for idx, (ll, ad) in enumerate(starts):
            if (time.time() - start_time) > TIME_LIMIT - 10.0:
                break
            remaining = TIME_LIMIT - (time.time() - start_time) - 8.0
            per_start = max(1.0, remaining / max(1, len(starts)))

            # L-BFGS-B refinement with current tau
            try:
                pts, sc = _optimize_from_params(
                    ll, ad, start_time, min(per_start * 0.7, 15.0), tau=tau
                )
                _update(pts, sc)
            except Exception:
                pass

            # Coordinate descent on min triangle
            if best_pts is not None:
                try:
                    pts2, sc2 = _coordinate_descent_min_triangle(
                        best_pts, start_time, min(per_start * 0.3, 5.0), max_rounds=500
                    )
                    _update(pts2, sc2)
                except Exception:
                    pass

    # Phase 2: Final polish on best with high tau
    if best_pts is not None and (time.time() - start_time) < TIME_LIMIT - 10.0:
        params = _points_to_edges(best_pts)
        if params is not None:
            ll, ad = params
            remaining = TIME_LIMIT - (time.time() - start_time) - 8.0
            try:
                pts, sc = _optimize_from_params(ll, ad, start_time, remaining * 0.4, tau=1000.0)
                _update(pts, sc)
            except Exception:
                pass
            if best_pts is not None:
                try:
                    pts2, sc2 = _coordinate_descent_min_triangle(
                        best_pts, start_time, remaining * 0.6, max_rounds=3000
                    )
                    _update(pts2, sc2)
                except Exception:
                    pass

    if best_pts is None:
        ll, ad = _regular_polygon_params()
        best_pts = _edges_to_points(ll, ad)

    return best_pts
