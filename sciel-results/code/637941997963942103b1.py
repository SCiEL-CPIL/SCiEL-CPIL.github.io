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

    # Compute minimum closeness of each edge to any tour edge
    # For each tour edge (a, b), for all (i,j):
    #   closeness = min(D[i,a] + D[j,b], D[i,b] + D[j,a])
    # We want min over all tour edges

    min_closeness = np.full((n, n), np.inf)

    for t in range(m):
        a = a_arr[t]
        b = b_arr[t]
        da = D[:, a]  # shape (n,)
        db = D[:, b]  # shape (n,)

        c1 = da[:, None] + db[None, :]  # D[i,a] + D[j,b]
        c2 = db[:, None] + da[None, :]  # D[i,b] + D[j,a]
        c = np.minimum(c1, c2)

        min_closeness = np.minimum(min_closeness, c)

    # Scale factor: use median of off-diagonal distances as typical edge length
    off_diag = D[~np.eye(n, dtype=bool)]
    scale = np.median(off_diag) if len(off_diag) > 0 else 0.1
    if scale <= 0:
        scale = 0.1

    # Normalize edge usage
    max_used = edge_n_used.max() if edge_n_used.size > 0 else 0
    if max_used > 0:
        norm_used = edge_n_used / max_used
    else:
        norm_used = np.zeros_like(edge_n_used)

    # Force field: decays with distance to nearest tour edge
    # beta=6 gives moderate repulsion zone for unit-square instances
    beta = 6.0
    force_field = np.exp(-beta * min_closeness)

    # Usage-based modulation: sqrt to encourage diversification
    usage_mod = np.sqrt(norm_used)

    # Additive penalty: scaled to instance's distance distribution
    # gamma controls overall penalty strength relative to typical edge length
    gamma = 0.15
    # alpha controls extra penalty for current tour edges
    alpha = 0.1

    # Base additive penalty from force field, modulated by usage
    penalty = gamma * scale * force_field * (0.3 + 0.7 * usage_mod)

    # Extra additive penalty for edges that are part of the current tour
    for t in range(m):
        a = a_arr[t]
        b = b_arr[t]
        penalty[a, b] += alpha * scale
        penalty[b, a] += alpha * scale

    # Combine: additive penalty on top of true distances
    result = D + penalty

    # Ensure symmetry and finite values
    result = (result + result.T) / 2.0
    np.fill_diagonal(result, 0.0)

    # Replace any non-finite values
    result = np.nan_to_num(result, nan=0.0, posinf=D.max() if D.max() > 0 else 1.0, neginf=0.0)

    return result
# EVOLVE-BLOCK-END
