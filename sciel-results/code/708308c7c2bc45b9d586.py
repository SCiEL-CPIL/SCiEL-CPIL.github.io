"""Heuristic matrix for ant colony optimization of TSP."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _nearest_neighbor_tour(dm, start=0):
    """Generate a tour using the nearest neighbor heuristic."""
    n = dm.shape[0]
    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    tour = [start]
    current = start
    for _ in range(n - 1):
        dists = dm[current].copy()
        dists[visited] = np.inf
        next_node = int(np.argmin(dists))
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
    return tour

def _tour_length(tour, dm):
    """Calculate the length of a closed tour."""
    n = len(tour)
    total = 0.0
    for i in range(n):
        total += dm[tour[i], tour[(i + 1) % n]]
    return total

def _two_opt(tour, dm, max_iter=1000):
    """Improve a tour using 2-opt local search."""
    n = len(tour)
    tour = list(tour)
    best_len = _tour_length(tour, dm)
    improved = True
    iters = 0
    while improved and iters < max_iter:
        improved = False
        iters += 1
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                # Calculate delta for reversing segment i+1..j
                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[(j + 1) % n]
                delta = (dm[a, c] + dm[b, d]) - (dm[a, b] + dm[c, d])
                if delta < -1e-10:
                    tour[i + 1:j + 1] = reversed(tour[i + 1:j + 1])
                    best_len += delta
                    improved = True
        # Early stop if no improvement in a full pass
        if not improved:
            break
    return tour, best_len

def _generate_consensus_heuristic(dm, time_budget=200.0, n_starts=20):
    """
    Generate a heuristic matrix based on the frequency of edges appearing
    in multiple diverse 2-opt optimized tours.
    """
    n = dm.shape[0]
    start_time = time.time()
    
    # Edge frequency matrix
    edge_freq = np.zeros((n, n), dtype=float)
    tour_lengths = []
    
    # Run multiple diverse local searches
    for s in range(n_starts):
        if time.time() - start_time > time_budget:
            break
            
        # Start from a random node to ensure diversity
        start_node = s % n
        tour = _nearest_neighbor_tour(dm, start=start_node)
        tour, length = _two_opt(tour, dm, max_iter=500)
        
        tour_lengths.append(length)
        
        # Count edges in the tour
        for i in range(n):
            u = tour[i]
            v = tour[(i + 1) % n]
            # Since TSP is symmetric, we can count both directions or just one.
            # The heuristic is applied to directed steps i->j.
            # If the tour uses u->v, it's a strong signal for that direction.
            # However, 2-opt tours are usually symmetric in quality.
            # We will count the directed edge as it appears in the tour.
            edge_freq[u, v] += 1.0
            
    # If no tours were generated (e.g., timeout immediately), fallback
    if len(tour_lengths) == 0:
        return 1.0 / (dm + 1e-9)
    
    # Normalize frequencies
    # The max frequency is n_starts (if all tours agree) or less.
    # We want to map [0, max_freq] to a range that boosts attractive edges.
    max_freq = np.max(edge_freq)
    if max_freq == 0:
        return 1.0 / (dm + 1e-9)
    
    # Heuristic: Base inverse distance scaled by consensus factor.
    # Consensus factor: 1.0 for edges not in any tour, up to 10.0 for edges in all tours.
    # Let's use a linear scaling: 1 + 9 * (freq / max_freq)
    consensus_factor = 1.0 + 9.0 * (edge_freq / max_freq)
    
    base_heuristic = 1.0 / (dm + 1e-9)
    heuristic = base_heuristic * consensus_factor
    
    return heuristic

def heuristics(distance_matrix):
    """
    Returns an edge-attractiveness matrix for ACO.
    Uses a consensus of multiple 2-opt tours to build the heuristic.
    """
    start_time = time.time()
    deadline = start_time + 250  # Safety margin before 300s limit
    
    dm = np.asarray(distance_matrix, dtype=float)
    n = dm.shape[0]
    
    if n == 0:
        return dm.copy()
    
    try:
        heuristic = _generate_consensus_heuristic(dm, time_budget=200.0, n_starts=20)
    except Exception:
        # Fallback to simple inverse distance if consensus generation fails
        heuristic = 1.0 / (dm + 1e-9)
    
    # Ensure no values are not positive or non-finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
