"""Guided local search penalty for TSP with model-and-optimize approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _two_opt(d, tour, max_passes=10):
    """Run 2-opt improvement on a tour using distance matrix d."""
    n = len(tour)
    tour = list(tour)
    
    for _ in range(max_passes):
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                a, b = tour[i], tour[i + 1]
                c, d2 = tour[j], tour[(j + 1) % n]
                # Check if 2-opt move improves
                if d[a][c] + d[b][d2] < d[a][b] + d[c][d2]:
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    improved = True
        if not improved:
            break
    return tour


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search penalty with additive penalties scaled to the
    instance's distance distribution, plus extra penalty on current-tour edges.
    """
    start_time = time.time()
    DEADLINE = 280

    n = edge_distance.shape[0]
    if n == 0:
        return edge_distance.copy()

    D = edge_distance

    # Extract tour edges
    tour_edges = []
    for i in range(n):
        a = local_opt_tour[i]
        b = local_opt_tour[(i + 1) % n]
        tour_edges.append((a, b))

    m = len(tour_edges)

    # Build arrays of tour edge endpoints
    a_arr = np.array([e[0] for e in tour_edges], dtype=int)
    b_arr = np.array([e[1] for e in tour_edges], dtype=int)

    # Compute min_closeness: for each pair (i,j), the minimum over all tour edges
    # of the "distance" from (i,j) to that tour edge in the metric space.
    min_closeness = np.full((n, n), np.inf)

    for t in range(m):
        a = a_arr[t]
        b = b_arr[t]
        da = D[:, a]
        db = D[:, b]

        c1 = da[:, None] + db[None, :]
        c2 = db[:, None] + da[None, :]
        c = np.minimum(c1, c2)

        min_closeness = np.minimum(min_closeness, c)

    # Normalize edge usage
    max_used = edge_n_used.max() if edge_n_used.size > 0 else 0
    if max_used > 0:
        norm_used = edge_n_used / max_used
    else:
        norm_used = np.zeros_like(edge_n_used)

    # Scale penalties to the instance's distance distribution
    # Use the median of non-zero distances as a scale reference
    nonzero_mask = D > 0
    if nonzero_mask.any():
        scale = np.median(D[nonzero_mask])
    else:
        scale = 1.0

    # Parameters
    beta = 5.0       # decay rate for the force field
    gamma = 2.0      # overall penalty strength (scaled by distance)
    alpha = 0.5      # extra penalty for current tour edges

    # Force field: decays with closeness to nearest tour edge
    force_field = np.exp(-beta * min_closeness / scale)

    # Usage-based modulation
    usage_mod = np.power(norm_used, 0.5)

    # Additive penalty scaled to distance distribution
    # The penalty is added to the true distance, not multiplied
    base_penalty = gamma * scale * force_field * (0.3 + 0.7 * usage_mod)

    # Extra penalty for edges that are part of the current tour
    # These have min_closeness ~ 0, so force_field ~ 1
    # Add an explicit boost to ensure they are strongly discouraged
    tour_mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        a = local_opt_tour[i]
        b = local_opt_tour[(i + 1) % n]
        tour_mask[a, b] = True
        tour_mask[b, a] = True

    extra_tour_penalty = alpha * scale * tour_mask

    # Combine: additive penalty
    penalty = base_penalty + extra_tour_penalty

    # Apply penalty to the distance matrix
    result = D + penalty

    # Ensure symmetry and finite values
    result = (result + result.T) / 2.0
    np.fill_diagonal(result, 0.0)

    # Replace any non-finite values
    result = np.nan_to_num(result, nan=0.0, posinf=D.max() if D.max() > 0 else 1.0, neginf=0.0)

    # Deadline check
    if time.time() - start_time > DEADLINE:
        return result

    return result
# EVOLVE-BLOCK-END
