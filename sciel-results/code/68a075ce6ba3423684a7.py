"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search perturbation: penalize edges that have been used
    in local optima, with a stronger emphasis on recently used edges.
    
    Strategy:
    1. Additive penalty proportional to usage count, scaled by mean distance
    2. Extra penalty for edges in the current tour (they are most likely to be removed)
    3. Smooth the penalty to avoid abrupt changes
    """
    start_time = time.time()
    deadline = start_time + 250.0  # Safety margin before 330s limit
    
    n = edge_distance.shape[0]
    mean_dist = np.mean(edge_distance)
    
    # Normalize usage counts for stability
    max_used = edge_n_used.max()
    if max_used > 0:
        norm_used = edge_n_used / (1.0 + max_used)
    else:
        norm_used = np.zeros_like(edge_n_used)
    
    # Base additive penalty: scaled by mean distance, weighted by usage
    # Use a moderate coefficient to perturb without destroying the structure
    penalty_weight = 0.25 * mean_dist
    penalized = edge_distance.copy()
    penalized += penalty_weight * edge_n_used
    
    # Additional penalty for edges in the current local optimum tour
    # These edges are the most 'stuck' and should be penalized more
    tour_penalty_weight = 0.15 * mean_dist
    for i in range(n):
        j = local_opt_tour[i % n]
        k = local_opt_tour[(i + 1) % n]
        # Penalize both directions
        penalized[i, j] += tour_penalty_weight
        penalized[j, i] += tour_penalty_weight
    
    # Ensure symmetry
    penalized = (penalized + penalized.T) / 2.0
    
    # Ensure finite values
    penalized = np.where(np.isfinite(penalized), penalized, edge_distance)
    
    # Ensure non-negative
    penalized = np.maximum(penalized, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        return edge_distance.copy()
    
    return penalized
# EVOLVE-BLOCK-END
