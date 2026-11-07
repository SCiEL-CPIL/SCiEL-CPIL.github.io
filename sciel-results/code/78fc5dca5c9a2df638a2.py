"""Guided local search penalty rule for the traveling salesman problem."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search perturbation for TSP.
    
    Strategy:
    1. Apply inverse-distance weighting to increase relative cost of short edges,
       forcing 2-opt to consider alternative (slightly longer) connections.
    2. Penalize edges in the current local-opt tour to force escape.
    3. Scale penalty by cumulative usage (edge_n_used) to target
       frequently-reused edges more strongly.
    4. Use a smooth, bounded penalty that preserves metric-like properties
       enough for 2-opt to work effectively.
    
    Key improvements:
      - Inverse-distance weighting as base perturbation
      - Adaptive usage-based penalty on tour edges
      - Smooth, bounded perturbation to keep distances well-conditioned
    """
    n = edge_distance.shape[0]
    
    # Time guard: ensure we finish well within 300s
    start_time = time.time()
    deadline = start_time + 280.0
    
    # Build mask of edges in the current local-opt tour
    tour_mask = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        a = local_opt_tour[i]
        b = local_opt_tour[(i + 1) % n]
        tour_mask[a, b] = 1.0
        tour_mask[b, a] = 1.0
    
    # Normalize usage with a saturating function to avoid extreme values
    max_used = np.max(edge_n_used)
    if max_used > 0:
        # Smooth saturating function: usage / (1 + usage)
        # Gives values in (0, 1) and saturates for large usage
        normalized_usage = edge_n_used / (1.0 + edge_n_used)
    else:
        normalized_usage = np.zeros_like(edge_n_used)
    
    # Compute a reference scale for inverse-distance weighting
    # Use median distance (excluding zeros/diagonal) as reference
    off_diag = edge_distance[~np.eye(n, dtype=bool)]
    ref_dist = np.median(off_diag[off_diag > 0])
    if ref_dist <= 0:
        ref_dist = 1.0
    
    # Inverse-distance weighting: 
    # For each edge, compute weight = ref_dist / (edge_distance + eps)
    # This increases relative cost of short edges, forcing exploration of longer alternatives
    eps = 1e-8
    inv_dist_weight = ref_dist / (edge_distance + eps)
    
    # Blend: use a moderate inverse-distance perturbation
    # alpha controls how much inverse-distance weighting is applied
    alpha = 0.15
    
    # Base perturbed distance: blend original with inverse-distance weighted
    # result = (1 - alpha) * original + alpha * inv_dist_weight * original
    # This preserves the relative ordering somewhat while boosting short edges
    perturbed = edge_distance * (1.0 + alpha * (inv_dist_weight - 1.0))
    
    # Ensure perturbed distances are positive and finite
    perturbed = np.maximum(perturbed, 1e-10)
    perturbed = np.nan_to_num(perturbed, nan=edge_distance, posinf=edge_distance, neginf=1e-10)
    
    # Adaptive base penalty for tour edges
    if max_used > 0:
        # Base penalty grows slowly with usage to encourage stronger escape later
        base_penalty = 0.08 + 0.015 * min(max_used, 15.0)
        # Usage boost: additional penalty scaled by normalized usage
        usage_boost = 0.20
    else:
        base_penalty = 0.08
        usage_boost = 0.20
    
    # Compute penalty only for tour edges
    # The penalty is applied multiplicatively
    tour_penalty = (base_penalty + usage_boost * normalized_usage) * tour_mask
    
    # Apply penalty multiplicatively to the perturbed matrix
    result = perturbed * (1.0 + tour_penalty)
    
    # Ensure diagonal is zero
    np.fill_diagonal(result, 0.0)
    
    # Final safety: ensure all values are finite and positive (off-diagonal)
    result = np.nan_to_num(result, nan=edge_distance, posinf=edge_distance, neginf=1e-10)
    off_diag_mask = ~np.eye(n, dtype=bool)
    result[off_diag_mask] = np.maximum(result[off_diag_mask], 1e-10)
    
    # Check deadline
    if time.time() > deadline:
        return result
    
    return result
# EVOLVE-BLOCK-END
