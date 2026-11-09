import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search perturbation for TSP.
    
    Strategy:
    - Strong calibrated penalty on edges in the current local optimum tour
    - Smooth tanh-scaled penalty on historically used edges
    - Degree-based topological bias
    - Edge-reversal bias to guide 2-opt toward promising reversals
    - Keep matrix finite and symmetric
    """
    start_time = time.time()
    deadline = start_time + 285  # 300s limit, leave 15s safety margin

    n = edge_distance.shape[0]

    # Work with float64 for numerical stability
    dist = edge_distance.astype(np.float64, copy=True)
    usage = edge_n_used.astype(np.float64, copy=True)

    # Make usage symmetric
    usage = 0.5 * (usage + usage.T)

    # Parse the tour
    tour = np.asarray(local_opt_tour)
    tour_len = len(tour)

    # Compute tour edge set and average tour edge length for calibration
    tour_edges = np.zeros((n, n), dtype=np.float64)
    tour_edge_lengths = []
    for i in range(tour_len):
        u = tour[i]
        v = tour[(i + 1) % tour_len]
        tour_edges[u, v] = 1.0
        tour_edges[v, u] = 1.0
        tour_edge_lengths.append(dist[u, v])

    tour_mean_edge = np.mean(tour_edge_lengths) if tour_edge_lengths else 1.0
    mean_dist = np.mean(dist)

    # --- 1. Calibrated tour penalty ---
    # Scale by the average tour edge length so the penalty is proportional
    # to what the tour "costs" on those edges
    tour_penalty_weight = 0.40 * tour_mean_edge
    dist += tour_penalty_weight * tour_edges

    # --- 2. Smooth tanh-scaled usage penalty ---
    max_usage = np.max(usage)
    if max_usage <= 0:
        max_usage = 1.0
    usage_scale = max(1.0, max_usage * 0.4)
    smooth_usage = np.tanh(usage / usage_scale)
    usage_penalty_weight = 0.12 * mean_dist
    dist += usage_penalty_weight * smooth_usage

    # --- 3. Degree-based topological bias ---
    degree = np.sum(usage > 0, axis=1)
    max_degree = np.max(degree)
    if max_degree <= 0:
        max_degree = 1.0
    norm_degree = degree / max_degree
    degree_bias = np.outer(norm_degree, norm_degree)
    degree_penalty_weight = 0.04 * mean_dist
    dist += degree_penalty_weight * degree_bias

    # --- 4. Edge-reversal bias ---
    # For each pair of tour edges, create a small negative bias on the
    # "cross" edges that would complete a 2-opt reversal.
    # Tour edges are (tour[i], tour[i+1]). A 2-opt reversal of segment
    # [i+1..j] removes (tour[i],tour[i+1]) and (tour[j],tour[j+1]),
    # adds (tour[i],tour[j]) and (tour[i+1],tour[j+1]).
    # We add a small negative bias to encourage 2-opt to consider these.
    reversal_bias = np.zeros((n, n), dtype=np.float64)
    # Use a limited window to keep it O(n*w) rather than O(n^2)
    window = min(30, tour_len // 3)
    for i in range(tour_len):
        a = tour[i]
        b = tour[(i + 1) % tour_len]
        for j in range(i + 2, min(i + 1 + window, tour_len)):
            c = tour[j]
            d = tour[(j + 1) % tour_len]
            # The new edges after reversal
            reversal_bias[a, c] -= 1.0
            reversal_bias[c, a] -= 1.0
            reversal_bias[b, d] -= 1.0
            reversal_bias[d, b] -= 1.0
    # Also handle wrap-around pairs
    for i in range(tour_len):
        a = tour[i]
        b = tour[(i + 1) % tour_len]
        for j in range(max(0, tour_len - window), tour_len):
            if j <= i:
                continue
            c = tour[j]
            d = tour[(j + 1) % tour_len]
            reversal_bias[a, c] -= 1.0
            reversal_bias[c, a] -= 1.0
            reversal_bias[b, d] -= 1.0
            reversal_bias[d, b] -= 1.0

    reversal_weight = 0.015 * mean_dist
    dist += reversal_weight * reversal_bias

    # Ensure symmetry
    dist = 0.5 * (dist + dist.T)

    # Ensure no negative values
    dist = np.maximum(dist, 0.0)

    # Ensure finite values
    dist = np.where(np.isfinite(dist), dist, mean_dist)

    # Zero out diagonal (self-loops shouldn't matter but keep clean)
    np.fill_diagonal(dist, 0.0)

    # Deadline check
    if time.time() > deadline:
        return dist

    return dist
