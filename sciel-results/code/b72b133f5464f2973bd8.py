"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Perturbation step for Guided Local Search with dynamic regret-based penalties.
    
    Args:
        edge_distance: (N, N) array of true Euclidean distances.
        local_opt_tour: (N,) array of node indices forming the current tour.
        edge_n_used: (N, N) array of symmetric counts of edge usage.
        
    Returns:
        (N, N) array of perturbed distances.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Leave 20s margin before 300s limit
    
    n = len(local_opt_tour)
    if n < 3:
        return edge_distance.copy()
    
    # Build the set of edges in the current tour
    # The tour is a closed cycle: tour[i] -> tour[i+1] -> ... -> tour[n-1] -> tour[0]
    tour_edges = set()
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        tour_edges.add((min(u, v), max(u, v)))
    
    # Calculate regret for each edge in the current tour
    # Regret = minimal increase in tour length if we remove this edge via a single 2-opt swap
    # For edge (u, v) in the tour, a 2-opt swap that removes it would replace edges (u,v) and (a,b)
    # with (u,a) and (v,b) (or (u,b) and (v,a))
    # The regret is the minimum over all valid 2-opt moves that break edge (u,v)
    
    # Precompute tour positions for O(1) lookup
    pos_in_tour = np.empty(n, dtype=int)
    for i in range(n):
        pos_in_tour[local_opt_tour[i]] = i
    
    # For each edge in the tour, compute its regret
    # An edge (u, v) is between positions pos_u and pos_u+1 (mod n) in the tour
    # A 2-opt move breaks edges (i, i+1) and (j, j+1) and reconnects as (i, j) and (i+1, j+1)
    # To break edge (u, v) which is at position k (connecting tour[k] to tour[k+1]),
    # we need a 2-opt move where one of the broken edges is (tour[k], tour[k+1])
    
    # For efficiency, we'll compute regret for each tour edge
    tour_edge_list = []
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        tour_edge_list.append((u, v, i))
    
    # Compute regret for each edge
    # Regret of edge at position i: min over j != i, j != i-1 (mod n) of
    #   [dist(tour[i], tour[j]) + dist(tour[i+1], tour[j+1]) - dist(tour[i], tour[i+1]) - dist(tour[j], tour[j+1])]
    # But we only consider moves that actually break this specific edge
    
    # Actually, for a 2-opt move breaking edges at positions i and j (i < j, j != i+1 mod n):
    # The length change = dist(tour[i], tour[j]) + dist(tour[i+1], tour[j+1]) - dist(tour[i], tour[i+1]) - dist(tour[j], tour[j+1])
    # (or the other pairing: dist(tour[i], tour[j+1]) + dist(tour[i+1], tour[j]) - dist(tour[i], tour[i+1]) - dist(tour[j], tour[j+1]))
    # We take the minimum of these two.
    
    # For edge at position i, its regret is the minimum length increase over all 2-opt moves that break edge i.
    # This means we consider all j != i, j != (i-1) mod n, j != (i+1) mod n
    
    # This is O(n^2) per edge, so O(n^3) total. For n=100, that's 10^6 operations, which is fine.
    
    regrets = np.zeros(n)
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        edge_len = edge_distance[u, v]
        
        min_increase = float('inf')
        for j in range(n):
            # Skip if j is i, i-1, or i+1 (mod n)
            if j == i or j == (i - 1) % n or j == (i + 1) % n:
                continue
            
            a = local_opt_tour[j]
            b = local_opt_tour[(j + 1) % n]
            edge_len_j = edge_distance[a, b]
            
            # Two possible reconnections
            # Option 1: (u, a) and (v, b)
            new_len1 = edge_distance[u, a] + edge_distance[v, b]
            delta1 = new_len1 - edge_len - edge_len_j
            
            # Option 2: (u, b) and (v, a)
            new_len2 = edge_distance[u, b] + edge_distance[v, a]
            delta2 = new_len2 - edge_len - edge_len_j
            
            delta = min(delta1, delta2)
            if delta < min_increase:
                min_increase = delta
        
        if min_increase == float('inf'):
            min_increase = 0.0
        regrets[i] = min_increase
    
    # Check time
    if time.time() > deadline:
        # Fallback: simple penalty
        max_used = edge_n_used.max()
        if max_used > 0:
            penalty = 0.1 * edge_n_used / (1.0 + max_used)
        else:
            penalty = np.zeros_like(edge_n_used)
        return edge_distance * (1.0 + penalty)
    
    # Normalize regrets: edges with higher regret are "harder" to break, so we penalize them more
    # Wait, actually we want to penalize edges that are easy to break? No, in GLS we penalize
    # edges that are frequently used in local optima to encourage exploring new solutions.
    # The regret-based approach: penalize edges that are "critical" (high regret means breaking them
    # costs a lot, so they're important to the current solution). We want to break them to explore.
    # So higher regret -> higher penalty.
    
    max_regret = regrets.max()
    if max_regret > 0:
        normalized_regrets = regrets / max_regret
    else:
        normalized_regrets = np.zeros(n)
    
    # Combine with usage count
    # Usage count: how often this edge has appeared in local optima
    # We want to penalize edges that are both frequently used AND have high regret
    usage_penalty = np.zeros((n, n))
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        # Symmetric
        usage_penalty[u, v] = normalized_regrets[i] * (1.0 + edge_n_used[u, v] * 0.5)
        usage_penalty[v, u] = usage_penalty[u, v]
    
    # Scale the penalty
    # The penalty should be significant but not dominate
    scale = 0.3
    perturbed = edge_distance * (1.0 + scale * usage_penalty)
    
    # Ensure symmetry
    perturbed = (perturbed + perturbed.T) / 2.0
    
    # Ensure no negative values and finite
    perturbed = np.maximum(perturbed, 0.0)
    perturbed = np.where(np.isfinite(perturbed), perturbed, edge_distance)
    
    return perturbed
# EVOLVE-BLOCK-END
