# EVOLVE-BLOCK-START
import numpy as np
import time


def _nearest_neighbor_tour(dist):
    n = dist.shape[0]
    visited = np.zeros(n, dtype=bool)
    tour = [0]
    visited[0] = True
    for _ in range(n - 1):
        cur = tour[-1]
        remaining = np.where(~visited)[0]
        if len(remaining) == 0:
            break
        d = dist[cur, remaining]
        nxt = remaining[np.argmin(d)]
        tour.append(nxt)
        visited[nxt] = True
    return tour


def _two_opt(dist, tour):
    n = len(tour)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                if j == i + 1:
                    continue
                a, b = tour[i - 1], tour[i]
                c, d = tour[j], tour[(j + 1) % n]
                # Current edges: a-b and c-d
                # New edges: a-c and b-d
                delta = dist[a, c] + dist[b, d] - dist[a, b] - dist[c, d]
                if delta < -1e-10:
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    improved = True
    return tour


def _compute_directed_bias(dist, nn_tour, k=5):
    n = dist.shape[0]
    # Start with squared inverse distance
    bias = 1.0 / (dist ** 2 + 1e-12)
    
    # Boost edges in the 2-opt NN tour
    for i in range(n):
        u = nn_tour[i]
        v = nn_tour[(i + 1) % n]
        bias[u, v] *= 5.0
    
    # Boost k-NN local edges
    for i in range(n):
        d = dist[i].copy()
        d[i] = np.inf
        k_nn = np.argsort(d)[:k]
        for j in k_nn:
            bias[i, j] *= 2.0
    
    return bias


def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    dist = np.asarray(distance_matrix, dtype=float)
    n = dist.shape[0]
    
    # Compute nearest neighbor tour
    nn_tour = _nearest_neighbor_tour(dist)
    
    # Apply 2-opt improvement
    if time.time() < deadline - 10.0:
        improved_tour = _two_opt(dist, nn_tour)
    else:
        improved_tour = nn_tour
    
    # Compute directed bias matrix
    bias = _compute_directed_bias(dist, improved_tour)
    
    # Ensure all values are positive and finite
    bias = np.where(np.isfinite(bias) & (bias > 0), bias, 1e-9)
    
    return bias
# EVOLVE-BLOCK-END
