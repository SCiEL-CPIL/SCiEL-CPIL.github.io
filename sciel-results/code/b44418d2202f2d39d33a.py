"""Guided local search penalty rule for TSP."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Perturbation step for Guided Local Search.
    
    Uses a 2-opt move regret model: for each edge in the current tour,
    compute the minimal length increase to remove it via a single 2-opt swap.
    Edges that are cheap to break (low regret) receive stronger penalties,
    guiding the perturbed 2-opt to preferentially break those edges.
    
    Args:
        edge_distance: (N, N) array of true Euclidean distances.
        local_opt_tour: (N,) array of node indices forming the current tour.
        edge_n_used: (N, N) array of symmetric counts of edge usage.
        
    Returns:
        (N, N) array of perturbed distances.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # 20s safety margin
    
    n = len(local_opt_tour)
    if n < 2:
        return edge_distance.copy()
    
    # Ensure we have a proper copy to work with
    perturbed = edge_distance.astype(np.float64).copy()
    
    # Get tour edges: (tour[i], tour[i+1]) for i in 0..n-2, and (tour[n-1], tour[0])
    # Build a mapping from edge to position in tour
    tour = np.asarray(local_opt_tour)
    
    # Compute regret for each tour edge
    # For edge (u, v) in the tour, a 2-opt swap removes edges (u,v) and (a,b)
    # and adds edges (u,a) and (v,b) or (u,b) and (v,a)
    # The regret is the minimum over all possible 2-opt swaps that include removing (u,v)
    # of (dist(u,a) + dist(v,b) - dist(u,v) - dist(a,b)) for the "cross" swap
    # or (dist(u,b) + dist(v,a) - dist(u,v) - dist(a,b)) for the "parallel" swap
    # Actually, in a 2-opt, we reverse a segment. If we remove edges (u,v) and (a,b),
    # we add (u,a) and (v,b) (reversing the segment between v and a).
    # The delta is dist(u,a) + dist(v,b) - dist(u,v) - dist(a,b)
    
    # For efficiency, we compute regret only for tour edges
    # Tour edges are (tour[i], tour[(i+1) % n])
    
    # Precompute: for each tour edge i, the endpoints
    # Let's index tour positions 0..n-1
    # Edge i connects tour[i] to tour[(i+1) % n]
    
    # For a given edge i (connecting u=tour[i], v=tour[(i+1)%n]),
    # a 2-opt that removes edge i and edge j (connecting a=tour[j], b=tour[(j+1)%n])
    # where j != i and j != (i-1)%n (adjacent edges can't be swapped in standard 2-opt)
    # The new edges would be (u, a) and (v, b) with the segment reversed.
    # Delta = dist(u,a) + dist(v,b) - dist(u,v) - dist(a,b)
    
    # We want the minimum delta over all valid j for each i.
    # This is O(n^2) per edge, O(n^3) total. For n=100, that's 10^6 operations, which is fine.
    
    # But we can be smarter. Let's compute for each pair of tour positions (i, j)
    # the delta of the 2-opt swap.
    
    # Actually, let's just do it directly for each tour edge.
    # For n=100, n^3 = 1,000,000 which is fast enough.
    
    regrets = np.zeros(n)
    
    # For each tour edge i
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        duv = edge_distance[u, v]
        
        min_delta = np.inf
        
        # For each other tour edge j
        for j in range(n):
            if j == i or j == (i - 1) % n:
                continue  # Can't swap with itself or adjacent edge
            
            a = tour[j]
            b = tour[(j + 1) % n]
            dab = edge_distance[a, b]
            
            # Two possible ways to reconnect:
            # Option 1: (u,a) and (v,b)
            delta1 = edge_distance[u, a] + edge_distance[v, b] - duv - dab
            # Option 2: (u,b) and (v,a)
            delta2 = edge_distance[u, b] + edge_distance[v, a] - duv - dab
            
            # In a proper 2-opt, we reverse the segment, so only one of these is valid
            # depending on the order. But for regret computation, we take the min of both
            # as the best possible single-swap improvement.
            
            delta = min(delta1, delta2)
            if delta < min_delta:
                min_delta = delta
        
        if np.isinf(min_delta):
            min_delta = 0.0
        regrets[i] = min_delta
    
    # Now compute penalties based on regret
    # Edges with low regret (easy to break) get higher penalty
    # Use: penalty = penalty_factor * (max_regret - regret) / (max_regret - min_regret + eps)
    
    max_regret = np.max(regrets)
    min_regret = np.min(regrets)
    regret_range = max_regret - min_regret + 1e-10
    
    # Normalize regret to [0, 1], where 1 = hardest to break, 0 = easiest
    normalized_regret = (regrets - min_regret) / regret_range
    
    # Penalty weight: higher for edges that are easy to break (low normalized_regret)
    # penalty_weight = 1 - normalized_regret, so easy-to-break edges get weight ~1
    penalty_weights = 1.0 - normalized_regret
    
    # Scale by penalty factor
    penalty_factor = 0.5
    
    # Also incorporate edge_n_used as a secondary signal
    # Edges used more often should get slightly higher penalty
    max_used = np.max(edge_n_used)
    if max_used > 0:
        used_norm = edge_n_used / max_used
        # Combine: primary is regret-based, secondary is usage-based
        combined_weights = 0.7 * penalty_weights + 0.3 * used_norm
    else:
        combined_weights = penalty_weights
    
    # Apply penalties to the tour edges
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        w = combined_weights[i]
        if w > 0:
            perturbed[u, v] += penalty_factor * w
            perturbed[v, u] += penalty_factor * w
    
    # Check deadline
    if time.time() > deadline:
        return perturbed
    
    # Ensure symmetry
    perturbed = (perturbed + perturbed.T) / 2.0
    
    # Ensure finite values
    if not np.all(np.isfinite(perturbed)):
        perturbed = np.nan_to_num(perturbed, nan=edge_distance, posinf=np.max(edge_distance), neginf=0.0)
    
    return perturbed
# EVOLVE-BLOCK-END
