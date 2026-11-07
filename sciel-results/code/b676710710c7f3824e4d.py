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

    # Stronger, smoothly saturating penalty:
    #   penalty = A * (1 - exp(-k * used / ref))
    # Apply the penalty ONLY to edges in the current local-optimum tour.
    # This keeps non-tour distances undistorted, so 2-opt is forced to
    # restructure the current tour rather than merely avoiding globally-used
    # edges.  The cap keeps the perturbation bounded so 2-opt on the modified
    # matrix remains well-behaved.
    k = 2.5
    A = 2.0

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

    # Compute penalty only on tour edges; zero elsewhere.
    penalty = np.zeros_like(D)
    if tour_mask.any():
        scaled = used[tour_mask] / ref
        p = A * (1.0 - np.exp(-k * scaled))
        p = np.clip(p, 0.0, A)
        penalty[tour_mask] = p

    # Multiply the true distances by (1 + penalty) for penalized edges.
    out = D * (1.0 + penalty)

    # Enforce exact symmetry.
    out = 0.5 * (out + out.T)

    if not np.all(np.isfinite(out)):
        base = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)
        out = 0.5 * (base + base.T)

    return out
# EVOLVE-BLOCK-END
