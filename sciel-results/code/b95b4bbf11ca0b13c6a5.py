"""Guided local search penalty rule for the traveling salesman problem."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search perturbation for TSP.
    
    Strategy:
    1. Penalize edges in the current local-opt tour to force escape.
    2. Scale penalty by cumulative usage (edge_n_used) to target
       frequently-reused edges more strongly.
    3. Use a smooth, bounded penalty that preserves metric-like properties
       enough for 2-opt to work effectively.
    4. Apply inverse-distance weighting: short edges (which are more valuable
       in a good tour) receive disproportionately larger penalties, creating
       a stronger gradient for escaping local optima.
    
    Key insight: In a good TSP tour, short edges are crucial. By penalizing
    short edges more heavily (inversely proportional to their length), we force
    the search to explore alternative connections, while long edges (which are
    less critical) receive smaller penalties.
    """
    n = edge_distance.shape[0]
    
    # Time guard: ensure we finish well within 300s
    start_time = time.time()
    deadline = start_time + 280.0
    
    # Build mask of edges in the current local-opt tour
    tour_mask = np.zeros((n, n), dtype=np.float64)
    
    # Collect tour edges
    tour_edges = set()
    for i in range(n):
        a = local_opt_tour[i]
        b = local_opt_tour[(i + 1) % n]
        tour_mask[a, b] = 1.0
        tour_mask[b, a] = 1.0
        tour_edges.add((min(a, b), max(a, b)))
    
    # Normalize usage with a smooth saturating function
    max_used = float(np.max(edge_n_used)) if edge_n_used.size > 0 else 0.0
    if max_used > 0:
        normalized_usage = edge_n_used / (1.0 + edge_n_used)
        quadratic_usage = normalized_usage ** 2
    else:
        normalized_usage = np.zeros_like(edge_n_used)
        quadratic_usage = np.zeros_like(edge_n_used)
    
    # Inverse-distance weighting: shorter edges get larger penalty multipliers
    # Use a smooth approximation: 1 / (distance + epsilon) where epsilon prevents division by zero
    # We want to penalize short edges more, so we use the reciprocal of the distance
    # Clip distances to avoid division by zero (diagonal is 0)
    safe_dist = np.where(edge_distance > 0, edge_distance, 1.0)
    # Inverse distance, normalized to have reasonable scale
    # The mean distance in unit square is ~0.5, so 1/d has mean ~2
    inv_dist = 1.0 / safe_dist
    # Normalize inv_dist to be in a reasonable range [0.5, 3] approximately
    inv_dist_min = np.min(inv_dist[tour_mask > 0]) if np.any(tour_mask > 0) else 1.0
    inv_dist_max = np.max(inv_dist[tour_mask > 0]) if np.any(tour_mask > 0) else 2.0
    if inv_dist_max > inv_dist_min:
        inv_dist_norm = (inv_dist - inv_dist_min) / (inv_dist_max - inv_dist_min)
        inv_dist_norm = 0.5 + inv_dist_norm  # Map to [0.5, 1.5]
    else:
        inv_dist_norm = np.ones_like(inv_dist)
    
    # Adaptive base penalty: start moderate, increase slightly with usage
    if max_used > 0:
        base_penalty = 0.08 + 0.008 * min(max_used, 20.0)
        usage_boost = 0.15
        quad_boost = 0.06 + 0.004 * min(max_used, 20.0)
        inv_dist_weight = 0.12 + 0.006 * min(max_used, 15.0)
    else:
        base_penalty = 0.08
        usage_boost = 0.15
        quad_boost = 0.06
        inv_dist_weight = 0.12
    
    # Compute penalty for tour edges with inverse-distance weighting
    # The penalty is: (base + usage terms) * (1 + inv_dist_weight * (inv_dist_norm - 1))
    # This means short edges (high inv_dist_norm) get extra penalty
    usage_penalty = usage_boost * normalized_usage + quad_boost * quadratic_usage
    inv_dist_factor = 1.0 + inv_dist_weight * (inv_dist_norm - 1.0)
    
    tour_penalty = (base_penalty + usage_penalty) * inv_dist_factor * tour_mask
    
    # Apply penalty additively (this is more stable than multiplicative for 2-opt)
    result = edge_distance + tour_penalty
    
    # Ensure finite values and positive distances
    result = np.nan_to_num(result, nan=edge_distance, posinf=edge_distance, neginf=0.0)
    
    # Ensure diagonal is zero (distance from node to itself)
    np.fill_diagonal(result, 0.0)
    
    # Check deadline
    if time.time() > deadline:
        return result
    
    return result
# EVOLVE-BLOCK-END
