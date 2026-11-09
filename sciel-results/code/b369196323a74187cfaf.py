# EVOLVE-BLOCK-START
import time

import numpy as np
from scipy.optimize import minimize


def _score_polygon(polygon):
    """
    Compute the per-vertex equidistance score for each vertex and return the
    minimum (the task metric). Returns 0.0 for invalid (non-strictly-convex or
    degenerate) configurations.

    The metric is scale/translation invariant, so we normalize first (subtract
    centroid, divide by mean distance to centroid).
    """
    n = polygon.shape[0]
    if n < 3:
        return 0.0

    # Normalize: subtract centroid, divide by mean distance to centroid.
    centroid = polygon.mean(axis=0)
    centered = polygon - centroid
    dists_to_centroid = np.linalg.norm(centered, axis=1)
    mean_dc = dists_to_centroid.mean()
    if mean_dc <= 0.01:
        return 0.0
    pts = centered / mean_dc

    # Strict convexity check: no repeated points, every turn nonzero and one sign,
    # total turning +-2*pi.
    # Edges
    edges = np.roll(pts, -1, axis=0) - pts
    edge_lens = np.linalg.norm(edges, axis=1)
    if np.any(edge_lens < 1e-9):
        return 0.0
    # Cross products of consecutive edges (signed turn).
    cross = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
    if np.any(np.abs(cross) < 1e-12):
        return 0.0
    # All turns must have the same sign.
    signs = np.sign(cross)
    if np.any(signs != signs[0]):
        return 0.0
    # Total turning should be +-2*pi.
    angles = np.arctan2(cross, edges[:, 0] * np.roll(edges[:, 0], -1) + edges[:, 1] * np.roll(edges[:, 1], -1))
    total_turn = angles.sum()
    if abs(abs(total_turn) - 2.0 * np.pi) > 1e-4:
        return 0.0

    # Per-vertex equidistance score.
    n_scores = np.ones(n)
    for i in range(n):
        # Distances from vertex i to the other n-1 vertices.
        d = np.linalg.norm(pts[i] - np.delete(pts, i, axis=0), axis=1)
        d_sorted = np.sort(d)
        # If any distance < 0.05, score 0.
        if d_sorted[0] < 0.05:
            n_scores[i] = 0.0
            continue
        # Find first index k minimizing d_{k+1} - d_k.
        diffs = np.diff(d_sorted)
        k = int(np.argmin(diffs))
        D = 0.5 * (d_sorted[k] + d_sorted[k + 1])
        if D < 0.1:
            n_scores[i] = 0.0
            continue
        # Choose four distances.
        if k == 0:
            sel = d_sorted[0:4]
        elif k == 17:
            sel = d_sorted[15:19]
        else:
            sel = d_sorted[k - 1:k + 3]
        # raw = mean over the four of max(d/D, D/d).
        ratio = np.maximum(sel / D, D / sel)
        raw = ratio.mean()
        n_scores[i] = min(1.0, 1.0 / raw)

    return float(n_scores.min())


def _softmin(values, temperature):
    """Smooth approximation to the minimum using log-sum-exp (negative)."""
    values = np.asarray(values, dtype=float)
    m = values.min()
    # Avoid overflow; values are in [0,1].
    return m - temperature * np.log(np.mean(np.exp((m - values) / temperature)))


def _per_vertex_scores(polygon):
    """Return the per-vertex score array (before taking the min)."""
    n = polygon.shape[0]
    centroid = polygon.mean(axis=0)
    centered = polygon - centroid
    dists_to_centroid = np.linalg.norm(centered, axis=1)
    mean_dc = dists_to_centroid.mean()
    if mean_dc <= 0.01:
        return np.zeros(n)
    pts = centered / mean_dc

    # Quick convexity: ensure turns are consistent (soft penalty handled by caller).
    n_scores = np.ones(n)
    for i in range(n):
        d = np.linalg.norm(pts[i] - np.delete(pts, i, axis=0), axis=1)
        d_sorted = np.sort(d)
        if d_sorted[0] < 0.05:
            n_scores[i] = 0.0
            continue
        diffs = np.diff(d_sorted)
        k = int(np.argmin(diffs))
        D = 0.5 * (d_sorted[k] + d_sorted[k + 1])
        if D < 0.1:
            n_scores[i] = 0.0
            continue
        if k == 0:
            sel = d_sorted[0:4]
        elif k == 17:
            sel = d_sorted[15:19]
        else:
            sel = d_sorted[k - 1:k + 3]
        ratio = np.maximum(sel / D, D / sel)
        raw = ratio.mean()
        n_scores[i] = min(1.0, 1.0 / raw)
    return n_scores


def _convexity_penalty(polygon):
    """
    Return a penalty (>= 0) that is 0 iff the polygon is strictly convex in the
    required sense, and large otherwise. This keeps the optimizer in a valid
    region without hard constraints.
    """
    pts = polygon
    edges = np.roll(pts, -1, axis=0) - pts
    edge_lens = np.linalg.norm(edges, axis=1)
    # Penalize near-zero edges (repeated points).
    min_edge = edge_lens.min()
    edge_pen = 0.0
    if min_edge < 1e-3:
        edge_pen = (1e-3 - min_edge) ** 2 * 1e6

    cross = edges[:, 0] * np.roll(edges[:, 1], -1) - edges[:, 1] * np.roll(edges[:, 0], -1)
    # Penalize turns near zero.
    min_abs_cross = np.abs(cross).min()
    turn_pen = 0.0
    if min_abs_cross < 1e-4:
        turn_pen = (1e-4 - min_abs_cross) ** 2 * 1e4

    # Penalize mixed signs of turns.
    signs = np.sign(cross)
    if np.any(signs != signs[0]):
        # Count violations.
        viol = np.sum(signs != signs[0])
        turn_pen += viol * 10.0

    # Total turning should be +-2*pi.
    angles = np.arctan2(cross, edges[:, 0] * np.roll(edges[:, 0], -1) + edges[:, 1] * np.roll(edges[:, 1], -1))
    total_turn = angles.sum()
    tt_pen = (abs(total_turn) - 2.0 * np.pi) ** 2 * 10.0

    return edge_pen + turn_pen + tt_pen


def _objective(params, temperature, convex_weight):
    """
    Objective to minimize: -softmin(per-vertex scores) + convexity penalty.
    Lower is better.
    """
    n = 20
    polygon = params.reshape(n, 2)
    scores = _per_vertex_scores(polygon)
    # softmin gives a smooth surrogate for the minimum (we want to maximize it,
    # so we minimize its negative).
    sm = _softmin(scores, temperature)
    pen = _convexity_penalty(polygon)
    return -sm + convex_weight * pen


def _regular_polygon(n=20, radius=1.0):
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return np.stack([x, y], axis=1)


def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a strictly convex 20-gon that maximizes the minimum per-vertex
    equidistance score. Starts from the regular polygon and refines with a
    smooth (soft-min) local optimization, then verifies with the exact metric.
    """
    n = 20
    deadline = time.time() + 600.0

    best_polygon = _regular_polygon(n, 1.0)
    best_score = _score_polygon(best_polygon)

    # Multi-start local optimization with a smooth surrogate.
    rng = np.random.default_rng(12345)
    temperature = 0.05
    convex_weight = 1.0

    n_starts = 1
    # Try a few perturbed starts to escape the regular-polygon plateau.
    starts = [best_polygon.copy()]
    for _ in range(n_starts - 1):
        perturbed = best_polygon + rng.normal(0.0, 0.05, size=(n, 2))
        starts.append(perturbed)

    for start in starts:
        if time.time() > deadline - 30.0:
            break
        params0 = start.reshape(-1)
        try:
            res = minimize(
                _objective,
                params0,
                args=(temperature, convex_weight),
                method="Nelder-Mead",
                options={
                    "maxiter": 4000,
                    "maxfev": 4000,
                    "xatol": 1e-8,
                    "fatol": 1e-10,
                    "adaptive": True,
                },
            )
            if time.time() > deadline - 30.0:
                break
            cand = res.x.reshape(n, 2)
            # Verify with the exact metric.
            s = _score_polygon(cand)
            if s > best_score:
                best_score = s
                best_polygon = cand
        except Exception:
            continue

    # Final high-precision polish pass on the best configuration found.
    if time.time() < deadline - 30.0:
        try:
            params0 = best_polygon.reshape(-1)
            res = minimize(
                _objective,
                params0,
                args=(0.01, 1.0),
                method="Nelder-Mead",
                options={
                    "maxiter": 2000,
                    "maxfev": 2000,
                    "xatol": 1e-10,
                    "fatol": 1e-12,
                    "adaptive": True,
                },
            )
            cand = res.x.reshape(n, 2)
            s = _score_polygon(cand)
            if s > best_score:
                best_score = s
                best_polygon = cand
        except Exception:
            pass

    # Safety: ensure the returned polygon is valid (finite, strictly convex).
    if not np.all(np.isfinite(best_polygon)):
        best_polygon = _regular_polygon(n, 1.0)

    return best_polygon


# EVOLVE-BLOCK-END
