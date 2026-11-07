"""Guided local search penalty rule for the TSP (direct approach)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE_S = 300.0
_START = None


def _elapsed():
    global _START
    if _START is None:
        _START = time.monotonic()
    return time.monotonic() - _START


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    # Deadline guard: if we ever approach the limit, return the safe base matrix.
    if _elapsed() > _DEADLINE_S - 30.0:
        return np.asarray(edge_distance, dtype=float).copy()

    D = np.asarray(edge_distance, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1] or D.shape[0] < 3:
        return D.copy()

    n = D.shape[0]

    # Guard against non-finite input; fall back to the (clean) base matrix.
    if not np.all(np.isfinite(D)):
        base = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)
        return base

    used = np.asarray(edge_n_used, dtype=float)
    if used.shape != D.shape:
        used = np.zeros_like(D)
    if not np.all(np.isfinite(used)):
        used = np.nan_to_num(used, nan=0.0, posinf=0.0, neginf=0.0)
    used = np.maximum(used, 0.0)

    # Reference scale: median of edge lengths of edges that have been used.
    if used.size:
        nz = used > 0.0
        if nz.any():
            ref = float(np.median(D[nz]))
        else:
            ref = float(np.median(D))
    else:
        ref = float(np.median(D))
    if not np.isfinite(ref) or ref <= 0.0:
        ref = 1.0

    # Build a mask of edges present in the current local-optimum tour.
    tour_mask = np.zeros_like(D, dtype=bool)
    t = local_opt_tour
    if t is not None and len(t) >= 3:
        for i in range(len(t)):
            a = int(t[i])
            b = int(t[(i + 1) % len(t)])
            if 0 <= a < n and 0 <= b < n and a != b:
                tour_mask[a, b] = True
                tour_mask[b, a] = True

    # Compute penalty with a smooth, saturating function.
    # Stronger penalty on tour edges, moderate on other used edges.
    # This creates an "exclusion zone" around the current tour while also
    # discouraging reuse of historically frequent (likely suboptimal) edges.
    
    # Parameters:
    # A_t: max penalty for tour edges (strong)
    # k_t: rate parameter for tour edges
    # A_h: max penalty for historical (non-tour) used edges (moderate)
    # k_h: rate parameter for historical edges
    A_t = 2.5
    k_t = 3.0
    A_h = 0.8
    k_h = 1.5

    penalty = np.zeros_like(D)

    # Historical penalty on all used edges (non-tour).
    hist_mask = (used > 0.0) & (~tour_mask)
    if hist_mask.any():
        scaled_h = used[hist_mask] / ref
        p_h = A_h * (1.0 - np.exp(-k_h * scaled_h))
        p_h = np.clip(p_h, 0.0, A_h)
        penalty[hist_mask] = p_h

    # Tour penalty (overwrites hist penalty on tour edges with stronger value).
    if tour_mask.any():
        scaled_t = used[tour_mask] / ref
        p_t = A_t * (1.0 - np.exp(-k_t * scaled_t))
        p_t = np.clip(p_t, 0.0, A_t)
        penalty[tour_mask] = p_t

    # Multiply the true distances by (1 + penalty) for penalized edges.
    out = D * (1.0 + penalty)

    # Enforce exact symmetry.
    out = 0.5 * (out + out.T)

    if not np.all(np.isfinite(out)):
        base = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)
        out = 0.5 * (base + base.T)

    return out
# EVOLVE-BLOCK-END
