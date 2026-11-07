"""Guided local search penalty rule for the traveling salesman problem."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search perturbation for TSP.
    
    Strategy:
    1. Multiplicative scaling on tour edges only: high penalty factor on
       current local-optimal tour edges to force 2-opt to sever them.
    2. Mild usage-based penalty on non-tour edges that have been used
       frequently in previous rounds, encouraging diversity.
    3. All other edges remain at true distances, preserving relative geometry.
    4. Returns a finite matrix of the same shape.
    """
    start_time = time.time()
    n = edge_distance.shape[0]
    
    # Deadline guard
    deadline = start_time + 280.0
    
    # Build tour edge mask
    tour_mask = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        a = local_opt_tour[i]
        b = local_opt_tour[(i + 1) % n]
        tour_mask[a, b] = 1.0
        tour_mask[b, a] = 1.0
    
    # High penalty factor for tour edges: scale them up significantly
    # This forces 2-opt to find alternative connections
    tour_penalty_factor = 1.8
    
    # Mild usage penalty for non-tour edges: only penalize edges that
    # have been used many times but are not in the current tour
    # Use a saturating function to keep it bounded
    non_tour_mask = 1.0 - tour_mask
    alpha_usage = 0.08
    usage_penalty = alpha_usage * edge_n_used / (1.0 + edge_n_used) * non_tour_mask
    
    # Combine: tour edges get high multiplicative penalty,
    # non-tour edges get mild usage-based penalty
    total_multiplier = 1.0 + tour_penalty_factor * tour_mask + usage_penalty
    
    # Apply multiplicatively
    result = edge_distance * total_multiplier
    
    # Ensure finite values
    result = np.nan_to_num(result, nan=edge_distance, posinf=edge_distance, neginf=0.0)
    
    # Final deadline check
    if time.time() > deadline:
        return result
    
    return result
# EVOLVE-BLOCK-END
