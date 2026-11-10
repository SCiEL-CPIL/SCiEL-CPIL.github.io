"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix):
    """
    Heuristic matrix for ACO TSP:
    1. Multi-start NN + 2-opt to find a high-quality reference tour.
    2. Base: inverse distance (4th power) for smooth long-edge suppression.
    3. Strong topological boost on reference tour edges.
    4. k-NN boost for local connectivity.
    5. Mild planarity penalty for very long edges.
    """
    start_time = time.time()
    deadline = start_time + 270.0

    n = distance_matrix.shape[0]
    dm = np.asarray(distance_matrix, dtype=np.float64)

    def tour_length(t):
        total = 0.0
        for i in range(n):
            total += dm[t[i], t[(i + 1) % n]]
        return total

    def nn_tour(start):
        visited = [False] * n
        tour = [start]
        visited[start] = True
        current = start
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
        return tour

    def two_opt(tour):
        t = list(tour)
        improved = True
        max_iter = 800
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
                    a, b = t[i - 1], t[i]
                    c, d = t[j], t[(j + 1) % n]
                    old_len = dm[a, b] + dm[c, d]
                    new_len = dm[a, c] + dm[b, d]
                    if new_len < old_len - 1e-10:
                        t[i:j + 1] = t[i:j + 1][::-1]
                        improved = True
                        if time.time() > deadline:
                            break
        return t

    # Multi-start NN + 2-opt for a better reference tour
    num_starts = 8
    best_tour = None
    best_len = np.inf

    # Always include start at 0
    starts = [0]
    # Add other strategic starts: nodes with low degree in NN graph
    for s in range(1, min(n, num_starts)):
        starts.append(s)

    for s in starts[:num_starts]:
        if time.time() > deadline - 5:
            break
        t = nn_tour(s)
        t = two_opt(t)
        tl = tour_length(t)
        if tl < best_len:
            best_len = tl
            best_tour = t

    tour = best_tour

    # Additional 2-opt refinement on the best tour
    if time.time() < deadline - 5:
        tour = two_opt(tour)

    # Build the heuristic matrix
    # Base: 4th power inverse distance
    safe_dist = np.where(dm == 0, 1e-9, dm)
    base_heuristic = 1.0 / (safe_dist ** 4)

    # Strong topological boost on reference tour edges
    tour_boost = 15.0
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        base_heuristic[u, v] *= tour_boost
        base_heuristic[v, u] *= tour_boost

    # k-NN boost (k=3)
    k = 3
    nn_boost = 3.0
    for i in range(n):
        dists = dm[i].copy()
        dists[i] = np.inf
        nn_indices = np.argsort(dists)[:k]
        for j in nn_indices:
            base_heuristic[i, j] *= nn_boost

    # Mild planarity penalty for very long edges
    if time.time() < deadline - 5:
        for i in range(n):
            dists_i = dm[i].copy()
            dists_i[i] = 0
            median_dist_i = np.median(dists_i)
            if median_dist_i > 1e-9:
                for j in range(n):
                    if i != j and dm[i, j] > 3.0 * median_dist_i:
                        base_heuristic[i, j] *= 0.5

    # Ensure no zeros, negatives, or non-finite values
    base_heuristic = np.where(np.isfinite(base_heuristic), base_heuristic, 1e-9)
    base_heuristic = np.maximum(base_heuristic, 1e-9)

    return base_heuristic
# EVOLVE-BLOCK-END
