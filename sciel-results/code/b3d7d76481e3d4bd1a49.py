"""Improved heuristic for ACO-TSP: smooth k-NN soft-min + consensus tour edge quality."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _two_opt(dm, tour):
    """In-place 2-opt improvement on a closed tour list (node indices)."""
    n = len(tour)
    if n < 4:
        return tour
    best = sum(dm[tour[i], tour[(i + 1) % n]] for i in range(n))
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            a = tour[i - 1]
            b = tour[i]
            for j in range(i + 1, n):
                c = tour[j]
                d = tour[(j + 1) % n]
                delta = dm[a, c] + dm[b, d] - dm[a, b] - dm[c, d]
                if delta < -1e-12:
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    best += delta
                    improved = True
    return tour


def _nearest_neighbor_tour(dm, start=0):
    """Greedy nearest-neighbor tour from `start`."""
    n = dm.shape[0]
    visited = [False] * n
    visited[start] = True
    tour = [start]
    cur = start
    for _ in range(n - 1):
        row = dm[cur].copy()
        row[visited] = np.inf
        nxt = int(np.argmin(row))
        visited[nxt] = True
        tour.append(nxt)
        cur = nxt
    return tour


def _soft_min_scores(dm, k=8, temperature=0.15):
    """
    Smooth surrogate: for each node i, compute a soft-min over its k nearest
    neighbors' distances, producing a per-edge quality score.
    
    For edge (i,j): quality = exp(-d(i,j) / (soft_min_i * temperature))
    This gives high scores to edges that are short relative to i's local neighborhood.
    """
    n = dm.shape[0]
    eps = 1e-12
    
    # For each node, get sorted distances to all other nodes
    # Use a vectorized approach: for each row, find k-th smallest distance
    # soft_min_i = k-th smallest distance from node i (excluding self)
    
    # Set diagonal to inf so it's excluded
    dm_work = dm.copy()
    np.fill_diagonal(dm_work, np.inf)
    
    # Get k-th smallest distance for each row (this is the "typical" local scale)
    # argsort is O(n^2 log n) which is fine for n=50
    sorted_dm = np.sort(dm_work, axis=1)  # shape (n, n), sorted along axis 1
    # k-th smallest is at index k-1 (0-indexed)
    kth_dist = sorted_dm[:, k - 1]  # shape (n,)
    
    # Ensure no zeros
    kth_dist = np.maximum(kth_dist, eps)
    
    # Soft-min score: for each pair (i,j), score = exp(-d(i,j) / (kth_dist[i] * temperature))
    # This is symmetric-ish but not exactly; we'll symmetrize
    scale_i = kth_dist[:, np.newaxis] * temperature  # shape (n, 1)
    scale_j = kth_dist[np.newaxis, :] * temperature  # shape (1, n)
    
    # Geometric mean of scales for symmetry
    scale = np.sqrt(scale_i * scale_j)  # shape (n, n)
    scale = np.maximum(scale, eps)
    
    # Soft-min quality: high when d is small relative to local scale
    quality = np.exp(-dm / scale)
    
    # Zero out diagonal
    np.fill_diagonal(quality, 0.0)
    
    return quality


def heuristics(distance_matrix):
    t0 = time.time()
    deadline = t0 + 280.0  # safety margin under 300s

    dm = np.asarray(distance_matrix, dtype=float)
    n = dm.shape[0]
    eps = 1e-9

    # Base: smooth soft-min based on k-NN local scale
    quality = _soft_min_scores(dm, k=8, temperature=0.15)
    
    # Consensus tour edge boost (smooth, not binary)
    if time.time() < deadline:
        tour = _nearest_neighbor_tour(dm, 0)
        tour = _two_opt(dm, tour)
        
        # Compute edge quality: for each tour edge, how "good" is it relative to alternatives?
        # Use the ratio of tour edge distance to the 2nd-shortest edge from each endpoint
        tour_edge_boost = np.zeros((n, n), dtype=float)
        
        for i in range(n):
            a = tour[i]
            b = tour[(i + 1) % n]
            d_ab = dm[a, b]
            
            # Get 2nd shortest distance from a (excluding self and b)
            row_a = dm[a].copy()
            row_a[a] = np.inf
            row_a[b] = np.inf
            if n > 2:
                second_shortest_a = np.sort(row_a)[0]  # shortest other than b
            else:
                second_shortest_a = d_ab
            
            row_b = dm[b].copy()
            row_b[b] = np.inf
            row_b[a] = np.inf
            if n > 2:
                second_shortest_b = np.sort(row_b)[0]
            else:
                second_shortest_b = d_ab
            
            # Edge is "good" if it's close to the best alternative
            # Ratio: d_ab / max(second_shortest_a, second_shortest_b, d_ab)
            # If d_ab is the shortest, ratio ~ 1; if much longer, ratio < 1
            ref = max(second_shortest_a, second_shortest_b, d_ab)
            ratio = d_ab / max(ref, eps)
            # Boost: higher ratio means edge is more competitive
            tour_edge_boost[a, b] = ratio
            tour_edge_boost[b, a] = ratio  # symmetric
        
        # Combine: quality * (1 + alpha * boost)
        # boost is in [0, 1], so this gently favors tour edges
        heuristic = quality * (1.0 + 3.0 * tour_edge_boost)
    else:
        heuristic = quality

    # Add a gentle global penalty for very long edges
    # Median edge length as reference
    upper_tri = dm[np.triu_indices(n, k=1)]
    med_len = np.median(upper_tri)
    # Penalize edges much longer than median: multiply by exp(-max(0, d/med - 1) * 0.5)
    long_penalty = np.exp(-np.maximum(0.0, dm / max(med_len, eps) - 1.0) * 0.5)
    heuristic = heuristic * long_penalty

    # Enforce positivity and finiteness
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    # Normalize so that the mean is 1 (prevents scale issues with ACO pheromone)
    mean_h = np.mean(heuristic)
    if mean_h > eps:
        heuristic = heuristic / mean_h

    return heuristic
# EVOLVE-BLOCK-END
