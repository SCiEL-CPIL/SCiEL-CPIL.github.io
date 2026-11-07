"""
Strategy: Zero-Lookahead Stochastic Perturbation
Instead of simulating 2-opt to find the best penalty, we apply a strong,
randomized penalty to the edges of the current tour. This forces the
evaluator's 2-opt (run on our returned matrix) to break the current local
optimum structure significantly. The subsequent 2-opt on true distances
then has a chance to find a better global region.
"""
import numpy as np
import time

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Generate a perturbation matrix by heavily penalizing edges in the current tour.
    The penalty magnitude is randomized to avoid getting stuck in a deterministic
    cycle of the same perturbation.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin
    
    n = len(local_opt_tour)
    
    # Create a copy of the true distance matrix
    perturbed = edge_distance.copy()
    
    # Extract edges of the current tour
    # We penalize these edges to discourage the next 2-opt from keeping them
    tour_edges = set()
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        if u > v:
            u, v = v, u
        tour_edges.add((u, v))
        
    # Strategy: 
    # 1. Identify edges in the current tour.
    # 2. Apply a strong multiplicative and additive penalty.
    #    The penalty should be high enough to break the local optimum but
    #    not so high that it creates a nonsensical landscape that 2-opt
    #    cannot navigate efficiently.
    #    
    # We use a random factor to ensure different runs/rounds explore different
    # directions. Since we don't know the "best" direction without lookahead,
    # we bet on "breaking the current structure strongly".
    
    # Base penalty scale
    # Typical edge length in unit square is ~0.3-0.5.
    # A penalty of 2.0-5.0 effectively removes these edges from consideration
    # for the first few 2-opt steps.
    
    # We can also use edge_n_used to bias the penalty: 
    # Edges used more often are more "stuck", so penalize them more.
    # However, simple uniform random penalty is often robust.
    
    # Let's try a combination:
    # Penalty = random_base + usage_factor * edge_n_used
    
    # Random base penalty between 1.5 and 4.0
    base_penalty = np.random.uniform(1.5, 4.0)
    
    # Usage weight: how much does history matter?
    # If an edge has been used 10 times, it's very likely part of the "trap".
    usage_weight = np.random.uniform(0.1, 0.5)
    
    for u, v in tour_edges:
        # Additive penalty
        penalty = base_penalty + usage_weight * edge_n_used[u, v]
        
        # Apply to both symmetric entries
        perturbed[u, v] += penalty
        perturbed[v, u] += penalty
        
        # Optional: Multiplicative boost to ensure dominance
        # If distance is small, additive might not be enough to dominate other edges.
        # Multiplying by a factor > 1 helps.
        # Let's add a small multiplicative factor for edges with high usage.
        if edge_n_used[u, v] > 5:
            perturbed[u, v] *= 1.2
            perturbed[v, u] *= 1.2

    # Check time (this operation is very fast, O(N) for N=100)
    if time.time() > deadline:
        # Fallback: return original if we somehow timed out (unlikely)
        return edge_distance

    return perturbed
