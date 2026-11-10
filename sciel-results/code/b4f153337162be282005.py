"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _nearest_neighbor_tour(dm):
    """Return a tour as a list of node indices starting at 0 using nearest neighbor."""
    n = dm.shape[0]
    visited = [False] * n
    tour = [0]
    visited[0] = True
    current = 0
    for _ in range(n - 1):
        # Find nearest unvisited node
        best_j = -1
        best_d = np.inf
        for j in range(n):
            if not visited[j]:
                d = dm[current, j]
                if d < best_d:
                    best_d = d
                    best_j = j
        tour.append(best_j)
        visited[best_j] = True
        current = best_j
    return tour


def _two_opt(tour, dm):
    """Improve a tour using 2-opt. Returns improved tour."""
    n = len(tour)
    if n < 4:
        return tour
    
    def tour_length(t):
        total = 0.0
        for i in range(n):
            total += dm[t[i], t[(i + 1) % n]]
        return total
    
    best_tour = tour[:]
    best_len = tour_length(best_tour)
    improved = True
    max_iters = 100
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                if i == 0 and j == n - 1:
                    continue
                # Calculate delta for reversing segment i..j
                a = best_tour[i - 1]
                b = best_tour[i]
                c = best_tour[j]
                d = best_tour[(j + 1) % n]
                delta = dm[a, c] + dm[b, d] - dm[a, b] - dm[c, d]
                if delta < -1e-10:
                    best_tour[i:j + 1] = best_tour[i:j + 1][::-1]
                    best_len += delta
                    improved = True
    return best_tour


def _compute_heuristic(distance_matrix):
    """Compute a heuristic matrix combining geometric and tour-based signals."""
    n = distance_matrix.shape[0]
    
    # 1. Base inverse distance (stronger power to suppress long edges)
    base = 1.0 / (distance_matrix + 1e-9) ** 2
    
    # 2. Nearest neighbor tour edges get a boost
    nn_tour = _nearest_neighbor_tour(distance_matrix)
    nn_tour = _two_opt(nn_tour, distance_matrix)
    
    tour_heuristic = np.ones_like(distance_matrix)
    for i in range(n):
        j = (i + 1) % n
        tour_heuristic[nn_tour[i], nn_tour[j]] = 5.0
        tour_heuristic[nn_tour[j], nn_tour[i]] = 5.0
    
    # 3. k-nearest neighbors get moderate boost
    k = 4
    knn_heuristic = np.ones_like(distance_matrix)
    for i in range(n):
        dists = distance_matrix[i].copy()
        dists[i] = np.inf
        # Get k nearest
        idx = np.argsort(dists)[:k]
        for j in idx:
            knn_heuristic[i, j] = 2.0
    
    # Combine: product of base and boosts
    heuristic = base * tour_heuristic * knn_heuristic
    
    # Ensure no zeros or non-finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic


def heuristics(distance_matrix):
    """Return an edge-attractiveness matrix for the given distance matrix."""
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 330s limit
    
    try:
        n = distance_matrix.shape[0]
        
        # Compute the heuristic
        heuristic = _compute_heuristic(distance_matrix)
        
        # Check time
        if time.time() > deadline:
            # Fallback to simple inverse distance
            return 1.0 / (distance_matrix + 1e-9)
        
        # Ensure valid output
        if not np.all(np.isfinite(heuristic)):
            heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
        if not np.all(heuristic > 0):
            heuristic = np.where(heuristic > 0, heuristic, 1e-9)
        
        return heuristic
        
    except Exception:
        # Fallback to simple inverse distance
        try:
            return 1.0 / (distance_matrix + 1e-9)
        except Exception:
            return np.ones_like(distance_matrix)
# EVOLVE-BLOCK-END
