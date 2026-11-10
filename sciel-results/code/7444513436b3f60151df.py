"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix):
    """
    Combined heuristic with aggressive long-edge suppression and topological guidance:
    1. Base: high-power inverse distance (5th power) to very aggressively penalize
       long/crossing edges, acting as a smooth "negative curvature" suppressor
       consistent with the planar nature of optimal 2D TSP tours.
    2. Topological boost: edges from a fast NN+2Opt reference tour are strongly boosted.
    3. k-NN boost: edges to nearest neighbors are boosted.
    4. Planarity-aware penalty: additional penalty for edges that are geometrically
       likely to cross (long chords relative to local neighborhood).
    This guides ACO ants toward structurally valid, short, non-crossing tours.
    """
    start_time = time.time()
    deadline = start_time + 275.0  # Safety margin for 300s limit

    n = distance_matrix.shape[0]
    dm = np.asarray(distance_matrix, dtype=np.float64)

    # 1. Generate a good reference tour using Nearest Neighbor
    visited = [False] * n
    tour = [0]
    visited[0] = True
    current = 0

    for _ in range(n - 1):
        min_dist = np.inf
        next_node = -1
        for j in range(n):
            if not visited[j]:
                d = dm[current, j]
                if d < min_dist:
                    min_dist = d
                    next_node = j
        tour.append(next_node)
        visited[next_node] = True
        current = next_node

    # 2. Improve with 2-Opt (limited iterations to save time)
    def tour_length(t):
        total = 0.0
        for i in range(n):
            total += dm[t[i], t[(i + 1) % n]]
        return total

    current_len = tour_length(tour)
    improved = True
    max_iter = 600
    iter_count = 0

    while improved and iter_count < max_iter:
        if time.time() > deadline:
            break
        improved = False
        iter_count += 1
        for i in range(n - 1):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if i == 0 and j == n - 1:
                    continue
                a, b = tour[i - 1], tour[i]
                c, d = tour[j], tour[(j + 1) % n]

                old_len = dm[a, b] + dm[c, d]
                new_len = dm[a, c] + dm[b, d]

                if new_len < old_len - 1e-10:
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    current_len -= old_len - new_len
                    improved = True
                    if time.time() > deadline:
                        break
        if time.time() > deadline:
            break

    # 3. Build the heuristic matrix
    # Base heuristic: 5th power inverse distance to very aggressively suppress
    # long-range/crossing edges (smooth negative-curvature penalty).
    safe_dist = np.where(dm == 0, 1e-9, dm)
    base_heuristic = 1.0 / (safe_dist ** 5)

    # Boost edges in the reference tour (strong topological guidance)
    tour_boost = 20.0
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        base_heuristic[u, v] *= tour_boost
        base_heuristic[v, u] *= tour_boost  # Symmetric boost for ACO

    # Boost k-NN edges (k=3)
    k = 3
    nn_boost = 4.0
    for i in range(n):
        dists = dm[i].copy()
        dists[i] = np.inf
        nn_indices = np.argsort(dists)[:k]
        for j in nn_indices:
            base_heuristic[i, j] *= nn_boost

    # Planarity-aware penalty: penalize edges that are very long relative to
    # the local neighborhood, discouraging crossing/chord edges.
    if time.time() < deadline - 10:
        for i in range(n):
            dists_i = dm[i].copy()
            dists_i[i] = 0
            median_dist_i = np.median(dists_i)
            if median_dist_i > 1e-9:
                for j in range(n):
                    if i != j and dm[i, j] > 2.5 * median_dist_i:
                        base_heuristic[i, j] *= 0.5  # Penalize long edges

    # Ensure no zeros, negatives, or non-finite values
    base_heuristic = np.where(np.isfinite(base_heuristic), base_heuristic, 1e-9)
    base_heuristic = np.maximum(base_heuristic, 1e-9)

    return base_heuristic
# EVOLVE-BLOCK-END
