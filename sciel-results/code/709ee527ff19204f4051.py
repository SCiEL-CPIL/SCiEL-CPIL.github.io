"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix):
    start_time = time.time()
    n = distance_matrix.shape[0]
    
    # Compute a strong reference tour using Greedy Nearest Neighbor + 2-Opt
    # This heuristic guides the ACO toward high-quality tour edges
    
    # Greedy Nearest Neighbor starting at 0
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    tour = [0]
    current = 0
    
    for _ in range(n - 1):
        dists = distance_matrix[current, :].copy()
        dists[visited] = np.inf
        next_node = np.argmin(dists)
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
    
    # 2-Opt improvement
    def tour_length(t):
        total = 0.0
        for i in range(len(t)):
            total += distance_matrix[t[i], t[(i + 1) % len(t)]]
        return total
    
    def two_opt(t):
        improved = True
        while improved:
            improved = False
            for i in range(len(t) - 1):
                for j in range(i + 2, len(t)):
                    if i == 0 and j == len(t) - 1:
                        continue
                    a, b = t[i], t[i + 1]
                    c, d = t[j], t[(j + 1) % len(t)]
                    delta = (distance_matrix[a, c] + distance_matrix[b, d]) - (distance_matrix[a, b] + distance_matrix[c, d])
                    if delta < 0:
                        t[i + 1:j + 1] = t[i + 1:j + 1][::-1]
                        improved = True
                        if time.time() - start_time > 280:
                            return t
        return t
    
    tour = two_opt(tour)
    
    # Build edge attractiveness matrix based on the reference tour
    # Edges in the reference tour get high attractiveness
    # Other edges get lower attractiveness based on distance
    
    # Base heuristic: inverse distance
    base = 1.0 / (distance_matrix + 1e-9)
    
    # Boost edges that are part of the reference tour
    for i in range(n):
        j = (i + 1) % n
        base[tour[i], tour[j]] = max(base[tour[i], tour[j]], 1000.0)
    
    # Ensure diagonal is not attractive
    np.fill_diagonal(base, 1e-9)
    
    # Ensure all values are positive
    base = np.maximum(base, 1e-9)
    
    return base
# EVOLVE-BLOCK-END
