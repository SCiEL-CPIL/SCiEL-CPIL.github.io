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


def _two_opt(tour, dm, max_iter=2000):
    """Improve a tour using 2-opt local search with early termination."""
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
                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[(j + 1) % n]
                delta = (dm[a, c] + dm[b, d]) - (dm[a, b] + dm[c, d])
                if delta < -1e-10:
                    tour[i + 1:j + 1] = reversed(tour[i + 1:j + 1])
                    best_len += delta
                    improved = True
    return tour, best_len


def _or_opt(tour, dm):
    """Improve a tour using Or-opt (relocating single nodes)."""
    n = len(tour)
    tour = list(tour)
    best_len = _tour_length(tour, dm)
    improved = True
    max_passes = 100
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(n):
            node = tour[i]
            # Remove node at position i
            a = tour[i - 1]
            b = tour[(i + 1) % n]
            tour.pop(i)
            n_minus_1 = n - 1
            # Try inserting at each position
            best_delta = 0.0
            best_pos = -1
            for j in range(n_minus_1):
                if j == i or j == (i - 1) % n_minus_1:
                    continue
                c = tour[j - 1]
                d = tour[j]
                # Original cost: ... a -> node -> b ...
                # New cost after removal: ... a -> b ...
                # Insertion cost: ... c -> node -> d ... replacing c -> d
                # delta = (dm[c,node] + dm[node,d] - dm[c,d]) + dm[a,b]
                delta = (dm[c, node] + dm[node, d] - dm[c, d]) + dm[a, b]
                if delta < best_delta - 1e-10:
                    best_delta = delta
                    best_pos = j
            if best_pos >= 0:
                tour.insert(best_pos, node)
                best_len += best_delta
                improved = True
            else:
                # Put it back
                tour.insert(i, node)
    return tour, best_len


def _run_local_search(dm, start_node, deadline):
    """Run a single instance of NN + 2-opt + Or-opt + 2-opt."""
    tour = _nearest_neighbor_tour(dm, start=start_node)
    tour, length = _two_opt(tour, dm, max_iter=500)
    tour, length = _or_opt(tour, dm)
    tour, length = _two_opt(tour, dm, max_iter=500)
    return tour, length


def heuristics(distance_matrix):
    """
    Returns an edge-attractiveness matrix for ACO.
    Uses a 'Multi-Elite' strategy: runs multiple local searches to find
    several high-quality tours, then constructs a heuristic matrix that
    boosts the frequency of edges appearing in these tours. This guides
    ants toward the 'consensus' good edges, which is often more robust
    than forcing them to follow a single specific tour.
    """
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 330s limit
    
    dm = np.asarray(distance_matrix, dtype=float)
    n = dm.shape[0]
    
    if n == 0:
        return dm.copy()
    
    # Base heuristic: inverse distance
    # We use a slightly softened inverse distance to avoid extreme values
    base_heuristic = 1.0 / (dm + 1e-6)
    
    # Initialize an accumulation matrix for edge frequencies
    edge_freq = np.zeros((n, n), dtype=float)
    
    # Define multiple starting points to diversify the initial NN tours
    # We want to find distinct local optima
    start_nodes = [0, n // 2, n // 3, n // 4, (n * 3) // 4, (n * 2) // 3, n // 5, (n * 4) // 5]
    
    # Ensure unique start nodes
    unique_starts = list(dict.fromkeys(start_nodes))
    
    # Run local search for each start node
    for start_node in unique_starts:
        if time.time() > deadline - 10: # Leave 10s buffer for final processing
            break
            
        try:
            tour, length = _run_local_search(dm, start_node, deadline)
            
            # Accumulate edge frequencies
            # The tour is a cycle: tour[0] -> tour[1] -> ... -> tour[n-1] -> tour[0]
            for i in range(n):
                u = tour[i]
                v = tour[(i + 1) % n]
                edge_freq[u, v] += 1.0
                # Note: TSP is usually undirected in distance, but ACO here uses directed probabilities.
                # However, the tour direction matters for pheromone deposition in the standard ACO model described.
                # The prompt says "adds 1/length to pheromone[i, j] for each step i to j of its closed tour, in travel direction only".
                # So we should only boost the directed edges in the direction of the tour.
                
        except Exception:
            continue
            
    # Check if we found any tours
    if np.sum(edge_freq) == 0:
        # Fallback to base heuristic
        return np.maximum(base_heuristic, 1e-9)
    
    # Construct the final heuristic matrix
    # Strategy: Combine base inverse distance with a strong boost for frequent edges.
    # We want edges that are in many good tours to be very attractive.
    
    # Normalize edge frequencies? Or just use them as a multiplier?
    # Let's use a multiplicative boost.
    # If an edge appears in K tours out of M runs, boost it.
    
    # To make it more robust, we can scale the boost based on the frequency.
    # boost = 1 + alpha * freq
    # But we want a strong signal. Let's try:
    # heuristic = base_heuristic * (1 + beta * edge_freq)
    
    # Alternatively, we can set the heuristic to be proportional to the frequency
    # but keep the base distance as a tie-breaker or floor.
    
    # Let's try: heuristic = base_heuristic * (1 + 10 * edge_freq)
    # This ensures that edges in the consensus are much more attractive.
    
    # However, if we just multiply, edges with 0 frequency get base_heuristic.
    # Edges with high frequency get a large boost.
    
    # Let's refine the boost factor.
    # If we have ~8 runs, max freq is 8.
    # boost_factor = 1 + 5 * edge_freq
    # Max boost ~ 41x.
    
    heuristic = base_heuristic * (1.0 + 5.0 * edge_freq)
    
    # Ensure no values are not positive or non-finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
