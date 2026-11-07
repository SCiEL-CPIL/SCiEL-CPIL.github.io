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

    # Hybrid penalty:
    # 1. Strong penalty on edges in the current local-optimum tour.
    # 2. Moderate penalty on other edges that have been used in the past.
    # This creates an "exclusion zone" that discourages 2-opt from accepting
    # swaps that reuse recently visited suboptimal edges, promoting exploration.
    
    # Parameters for tour edges (stronger repulsion)
    A_tour = 2.5
    k_tour = 3.0
    
    # Parameters for historical edges (moderate repulsion)
    A_hist = 0.5
    k_hist = 1.5

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

    # Identify edges that have been used (historical mask)
    hist_mask = used > 0.0
    
    # Initialize penalty matrix
    penalty = np.zeros_like(D)
    
    # Apply historical penalty to all used edges
    if hist_mask.any():
        scaled_hist = used[hist_mask] / ref
        p_hist = A_hist * (1.0 - np.exp(-k_hist * scaled_hist))
        p_hist = np.clip(p_hist, 0.0, A_hist)
        penalty[hist_mask] = p_hist
        
    # Overwrite with stronger tour penalty for edges in the current tour
    if tour_mask.any():
        scaled_tour = used[tour_mask] / ref
        p_tour = A_tour * (1.0 - np.exp(-k_tour * scaled_tour))
        p_tour = np.clip(p_tour, 0.0, A_tour)
        penalty[tour_mask] = p_tour

    # Multiply the true distances by (1 + penalty) for penalized edges.
    out = D * (1.0 + penalty)

    # Enforce exact symmetry.
    out = 0.5 * (out + out.T)

    if not np.all(np.isfinite(out)):
        base = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)
        out = 0.5 * (base + base.T)

    return out
# EVOLVE-BLOCK-END
