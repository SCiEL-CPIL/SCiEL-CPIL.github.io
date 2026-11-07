# EVOLVE-BLOCK-START
import numpy as np


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search perturbation using a repulsive force model.
    
    The penalty for each edge is inversely proportional to the Euclidean distance,
    so short, high-value edges receive disproportionately large penalties. This
    forces the search to explore alternative short connections and escape local
    optima.
    
    Design:
    1. Inverse-distance weighting: short edges get larger relative penalties
    2. Usage-count modulation: frequently used edges get amplified penalties
    3. Smooth, bounded nonlinearity: ensures numerical stability while
       maintaining strong gradient for escaping local optima
    """
    n = edge_distance.shape[0]
    
    # Ensure we have valid inputs
    if edge_distance is None or edge_distance.shape[0] != edge_distance.shape[1]:
        return edge_distance.copy()
    
    # Create a copy to avoid modifying the input
    penalized = edge_distance.copy()
    
    # Set diagonal to 0 (distance from a point to itself)
    np.fill_diagonal(penalized, 0.0)
    
    # Compute inverse distance weighting
    # Short edges (small d) -> large 1/d -> larger penalty multiplier
    # Long edges (large d) -> small 1/d -> smaller penalty multiplier
    eps = 1e-6
    inv_dist = 1.0 / (edge_distance + eps)
    np.fill_diagonal(inv_dist, 0.0)
    
    # Normalize inverse distances to [0, 1]
    inv_dist_max = inv_dist.max()
    if inv_dist_max > 0:
        inv_dist_norm = inv_dist / inv_dist_max
    else:
        inv_dist_norm = np.zeros_like(inv_dist)
    
    # Usage count normalization
    usage = edge_n_used.astype(float)
    usage_max = usage.max()
    if usage_max > 0:
        usage_norm = usage / usage_max
    else:
        usage_norm = np.zeros_like(usage)
    
    # Combined penalty weight: product of usage and geometric factor
    # This creates a repulsive force that is strongest for edges that are
    # both short (high value) and frequently used (part of local optima)
    geometric_factor = 1.0 + 3.0 * inv_dist_norm
    combined_weight = usage_norm * geometric_factor
    
    # Apply a smooth, bounded nonlinearity to amplify the effect
    # Use tanh to keep values in a reasonable range while maintaining
    # strong gradient for frequently-used short edges
    # Scale combined_weight to enhance the effect
    scaled_weight = 2.0 * combined_weight
    nonlinear_factor = np.tanh(scaled_weight)
    
    # Penalty multiplier: base + amplified component
    # The multiplier is at least 1.0 (no reduction) and can be up to ~1+max_penalty
    max_penalty = 2.5  # Maximum additional multiplier
    penalty_multiplier = 1.0 + max_penalty * nonlinear_factor
    
    # Apply the penalty to the distance matrix
    penalized = edge_distance * penalty_multiplier
    
    # Ensure the diagonal remains 0
    np.fill_diagonal(penalized, 0.0)
    
    # Ensure symmetry
    penalized = 0.5 * (penalized + penalized.T)
    
    # Ensure all values are finite and non-negative
    penalized = np.where(np.isfinite(penalized), penalized, 0.0)
    penalized = np.maximum(penalized, 0.0)
    
    return penalized
# EVOLVE-BLOCK-END
