# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search perturbation using a repulsive force model.
    
    The penalty for each edge is designed to be inversely proportional to the
    Euclidean distance (short edges get larger relative penalties), which creates
    a strong gradient to escape local optima by forcing the search to consider
    alternative short connections that were previously underutilized or part of
    the current local optimum.
    
    We combine:
    1. A base penalty proportional to usage count (edge_n_used)
    2. A geometric weighting that amplifies penalties for short edges
    3. A nonlinear scaling to ensure the perturbation is strong enough to
       disrupt the current local optimum
    """
    n = edge_distance.shape[0]
    
    # Ensure we have valid inputs
    if edge_distance is None or edge_distance.shape[0] != edge_distance.shape[1]:
        return edge_distance.copy()
    
    # Create a copy to avoid modifying the input
    penalized = edge_distance.copy()
    
    # Extract the diagonal (distance 0) and set it to 0 in the penalty matrix
    np.fill_diagonal(penalized, 0.0)
    
    # Compute the inverse distance weighting (repulsive force model)
    # For short edges, 1/d is large, so they get amplified penalties
    # For long edges, 1/d is small, so they get smaller relative penalties
    # Add a small epsilon to avoid division by zero
    eps = 1e-8
    inv_dist = 1.0 / (edge_distance + eps)
    np.fill_diagonal(inv_dist, 0.0)
    
    # Normalize inverse distances to [0, 1] range for consistent scaling
    inv_dist_max = inv_dist.max()
    if inv_dist_max > 0:
        inv_dist_norm = inv_dist / inv_dist_max
    else:
        inv_dist_norm = np.zeros_like(inv_dist)
    
    # Base penalty from usage count
    # Use a smooth, bounded function to avoid extreme penalties
    usage = edge_n_used.astype(float)
    usage_max = usage.max()
    if usage_max > 0:
        usage_norm = usage / usage_max
    else:
        usage_norm = np.zeros_like(usage)
    
    # Combine usage with geometric weighting
    # The penalty is larger for edges that are both frequently used AND short
    # This creates a repulsive force that pushes the search away from the
    # current local optimum's preferred short edges
    combined_penalty = usage_norm * (1.0 + 2.0 * inv_dist_norm)
    
    # Apply nonlinear scaling to strengthen the perturbation
    # Use a power law to amplify the effect
    alpha = 1.5
    penalty_factor = 1.0 + 0.5 * np.power(combined_penalty, alpha)
    
    # Apply the penalty to the distance matrix
    penalized = edge_distance * penalty_factor
    
    # Ensure the diagonal remains 0
    np.fill_diagonal(penalized, 0.0)
    
    # Ensure symmetry
    penalized = 0.5 * (penalized + penalized.T)
    
    # Ensure all values are finite and non-negative
    penalized = np.where(np.isfinite(penalized), penalized, 0.0)
    penalized = np.maximum(penalized, 0.0)
    
    return penalized
# EVOLVE-BLOCK-END
