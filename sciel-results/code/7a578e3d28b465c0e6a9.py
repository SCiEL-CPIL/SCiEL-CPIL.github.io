"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search penalty: penalize edges that have been used in previous
    local optima, using an adaptive penalty with edge-specific decay based on
    both usage frequency and edge length.
    """
    n = len(local_opt_tour)
    max_usage = edge_n_used.max()
    
    if max_usage == 0:
        return edge_distance.copy()
    
    # Normalize usage to [0, 1] range
    norm_usage = edge_n_used / max_usage
    
    # Adaptive penalty: stronger for frequently-used edges, but also consider
    # that longer edges are less "valuable" to preserve, so penalize them more
    # Use a power law on usage for sharper differentiation
    # and scale penalty inversely with edge length (shorter edges get less penalty)
    
    # Penalty strength based on usage with power curve
    # Higher power = sharper distinction between used and unused edges
    gamma = 1.5  # power exponent for usage
    
    # Base penalty from usage
    base_penalty = norm_usage ** gamma
    
    # Edge-length-aware scaling: shorter edges are more important to preserve
    # so they should get less penalty. Use normalized edge length.
    # Create a mask for the diagonal (self-loops) to avoid division issues
    diag_mask = np.eye(n, dtype=bool)
    edge_lengths = edge_distance.copy()
    
    # Normalize edge lengths to [0, 1] for scaling
    min_len = edge_lengths[~diag_mask].min()
    max_len = edge_lengths[~diag_mask].max()
    range_len = max_len - min_len
    
    if range_len > 0:
        norm_lengths = (edge_lengths - min_len) / range_len
    else:
        norm_lengths = np.zeros_like(edge_lengths)
    
    # Shorter edges (norm_lengths near 0) get less penalty multiplier
    # Longer edges (norm_lengths near 1) get more penalty multiplier
    # Use a linear scaling: penalty_multiplier = 1 + delta * norm_length
    delta_len = 0.5  # maximum additional penalty for longest edges
    
    penalty_multiplier = 1.0 + delta_len * norm_lengths
    
    # Combined penalty factor
    alpha = 0.25  # maximum penalty factor for most-used edges
    
    penalty_factor = 1.0 + alpha * base_penalty * penalty_multiplier
    
    # Ensure the matrix remains finite and same shape
    result = edge_distance * penalty_factor
    
    # Safety: ensure no NaN/inf
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=edge_distance, posinf=edge_distance, neginf=edge_distance)
    
    return result
# EVOLVE-BLOCK-END
