"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search penalty: penalize edges that have been used in previous
    local optima, using a smooth adaptive penalty based on usage frequency.
    
    The penalty is stronger for edges that appear more frequently across iterations,
    using a smooth saturation function to avoid abrupt changes.
    """
    n = len(local_opt_tour)
    max_usage = edge_n_used.max()
    
    if max_usage == 0:
        return edge_distance.copy()
    
    # Normalize usage to [0, 1] range
    norm_usage = edge_n_used / max_usage
    
    # Use a smooth penalty: 1 + alpha * (1 - exp(-beta * norm_usage))
    # This gives a gentle penalty for lightly-used edges and stronger for heavily-used ones
    alpha = 0.15  # maximum penalty factor (15% increase for most-used edges)
    beta = 2.0    # controls how quickly the penalty saturates
    
    penalty_factor = 1.0 + alpha * (1.0 - np.exp(-beta * norm_usage))
    
    # Ensure the matrix remains finite and same shape
    result = edge_distance * penalty_factor
    
    # Safety: ensure no NaN/inf
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=edge_distance, posinf=edge_distance, neginf=edge_distance)
    
    return result
# EVOLVE-BLOCK-END
