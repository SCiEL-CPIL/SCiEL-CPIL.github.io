"""Guided local search penalty for TSP: topological disruption of over-long tour edges."""
import numpy as np
import time

_DEADLINE = time.perf_counter() + 310.0  # safety margin before the 330s kill


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    n = edge_distance.shape[0]
    if n <= 2:
        return edge_distance.copy()

    # True distances (always input is the true matrix)
    d = np.asarray(edge_distance, dtype=float)

    # Build the edge list of the current tour (closed cycle)
    tour = np.asarray(local_opt_tour, dtype=int)
    # Ensure length == n and is a permutation; fallback to identity if malformed
    if tour.shape[0] != n:
        tour = np.arange(n)

    # Edge endpoints for the closed tour
    i = tour[:-1]
    j = tour[1:]
    # closing edge
    i = np.append(i, tour[0])
    j = np.append(j, tour[-1])

    # Lengths of each tour edge (true)
    tour_len = d[i, j]

    # Local geometric reference: for each edge (a,b), the cheapest edge from a
    # and from b (excluding the edge itself) gives a "should be about this long" scale.
    # Use the 2nd-smallest incident distance as a robust local scale.
    # Compute per-node local scale: the minimum distance to any other node.
    # To avoid the tour edge itself dominating, we use the two smallest incident edges.
    # Efficient approach: find the two smallest off-diagonal entries per row.
    # Mask diagonal with a large value, then take two smallest.
    diag_mask = np.eye(n, dtype=bool)
    d_work = d.copy()
    d_work[diag_mask] = np.inf

    # two smallest per row
    # argsort is O(n^2 log n) which is fine for n=100
    order = np.argsort(d_work, axis=1)
    min1 = d_work[np.arange(n), order[:, 0]]
    min2 = d_work[np.arange(n), order[:, 1]]
    local_scale = np.maximum(min1, min2)  # a robust "expected short edge" scale
    # Guard against zero scale
    local_scale = np.where(local_scale > 1e-12, local_scale, 1.0)

    # Over-length ratio for each tour edge: how much longer than local expectation
    ratio = tour_len / (local_scale[i] + local_scale[j])  # normalize by both endpoints
    # ratio ~1 means the edge is about as long as a typical short local edge.
    # Larger ratio => more "wasteful" edge that 2-opt tends to keep if neighbors are fixed.

    # We want to penalize edges with the largest ratios (the structurally bad ones).
    # Rank the tour edges by ratio and penalize the top fraction heavily.
    k = max(1, int(round(0.25 * n)))  # top 25% of tour edges by over-length
    # indices of the k largest ratios
    top_idx = np.argsort(ratio)[-k:]

    # Build a penalty matrix (additive), zero by default
    penalty = np.zeros((n, n), dtype=float)

    # Base usage-based mild penalty for all edges seen before (symmetric)
    usage = np.asarray(edge_n_used, dtype=float)
    usage = np.maximum(usage, 0.0)
    # Use smooth saturation: u/(1+u) which is bounded in [0,1) and smooth
    usage_scaled = usage / (1.0 + usage)
    # Add a small uniform-ish penalty weighted by usage (symmetric)
    penalty += 0.08 * usage_scaled

    # Topological disruption: heavy additive penalty on the worst tour edges.
    # Magnitude scales with the local edge length so it's meaningful in distance units.
    for idx in top_idx:
        a, b = i[idx], j[idx]
        # penalty magnitude: enough to push this edge out of the 2-opt neighborhood
        # Use a fraction of the tour's average edge length so it's comparable.
        mag = 0.6 * tour_len[idx] + 0.3 * np.mean(tour_len)
        penalty[a, b] += mag
        penalty[b, a] += mag

    # Symmetrize to be safe
    penalty = 0.5 * (penalty + penalty.T)

    # Result: true distance + penalty. Keep finite.
    result = d + penalty
    # Ensure finiteness (defensive)
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=0.0)

    return result
