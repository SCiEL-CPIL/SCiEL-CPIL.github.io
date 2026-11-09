import numpy as np
import time

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 50.0  # Safe margin
    
    D = np.asarray(distance_matrix, dtype=float)
    n = D.shape[0]
    
    if n < 2:
        return np.ones_like(D)

    def tour_length(tour):
        return float(np.sum(D[tour, np.roll(tour, -1)]))

    def two_opt(tour):
        tour = list(tour)
        improved = True
        while improved:
            improved = False
            for i in range(n - 1):
                for j in range(i + 2, n):
                    if i == 0 and j == n - 1:
                        continue
                    a, b = tour[i], tour[i+1]
                    c, d = tour[j], tour[(j+1) % n]
                    delta = D[a, c] + D[b, d] - D[a, b] - D[c, d]
                    if delta < 1e-10:
                        tour[i+1:j+1] = reversed(tour[i+1:j+1])
                        improved = True
        return tour

    def or_opt(tour):
        tour = list(tour)
        improved = True
        while improved:
            improved = False
            for size in [1, 2, 3]:
                if n < size + 2:
                    continue
                for i in range(n - size):
                    if i + size > n:
                        continue
                    seg = tour[i : i+size]
                    prev_i = tour[(i-1) % n]
                    next_i = tour[(i+size) % n]
                    orig_cost = D[prev_i, seg[0]] + D[seg[-1], next_i]
                    
                    remaining = tour[:i] + tour[i+size:]
                    if not remaining:
                        continue
                        
                    best_delta = 1e-10
                    best_pos = -1
                    
                    for k in range(len(remaining)):
                        prev_k = remaining[(k-1) % len(remaining)]
                        next_k = remaining[k % len(remaining)]
                        new_cost = D[prev_k, seg[0]] + D[seg[-1], next_k]
                        delta = new_cost - orig_cost
                        if delta < best_delta:
                            best_delta = delta
                            best_pos = k
                            
                    if best_pos != -1:
                        tour = remaining[:best_pos] + seg + remaining[best_pos:]
                        improved = True
        return tour

    best_tour = None
    best_len = float('inf')
    
    for start_node in range(n):
        if time.time() > deadline - 5.0:
            break
            
        tour = [start_node]
        visited = np.zeros(n, dtype=bool)
        visited[start_node] = True
        current = start_node
        
        for _ in range(n - 1):
            dists = D[current].copy()
            dists[visited] = np.inf
            next_node = np.argmin(dists)
            tour.append(next_node)
            visited[next_node] = True
            current = next_node
            
        tour = two_opt(tour)
        tour = or_opt(tour)
        tour = two_opt(tour)
        
        curr_len = tour_length(tour)
        if curr_len < best_len:
            best_len = curr_len
            best_tour = tour

    if best_tour is None:
        best_tour = list(range(n))

    heuristic = 1.0 / (D + 1e-9)
    np.fill_diagonal(heuristic, 0.0)
    
    boost_factor = 100.0
    for i in range(n):
        u = best_tour[i]
        v = best_tour[(i + 1) % n]
        heuristic[u, v] *= boost_factor
        heuristic[v, u] *= boost_factor
        
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
