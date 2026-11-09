"""Rewritten guided local search penalty for TSP."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search perturbation for TSP.
    
    Strategy:
    - Rank-based soft constraint: sort edges by usage count and apply a
      convex (quadratic) penalty based on normalized rank. This creates a
      steep barrier against the most frequently used edges while leaving
      low-usage edges nearly unperturbed.
    - Strong penalty on current tour edges to escape local optimum.
    - Smooth, bounded penalties to maintain numerical stability.
    
    Returns a finite distance matrix of the same shape.
    """
    n = edge_distance.shape[0]
    
    # Build a penalty matrix
    penalty = np.zeros_like(edge_distance, dtype=np.float64)
    
    # 1) Strong penalty on edges in the current local-optimal tour
    for i in range(n):
        j = local_opt_tour[i]
        penalty[i, j] = 1.0
        penalty[j, i] = 1.0
    
    # 2) Rank-based usage penalty:
    # Flatten the upper triangle of edge_n_used, sort, and assign ranks
    # Apply a convex (quadratic) function of normalized rank
    
    # Extract upper triangle values (excluding diagonal)
    triu_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    usage_values = edge_n_used[triu_mask]
    
    # Sort usage values to get ranks
    sorted_usage = np.sort(usage_values)
    m = len(sorted_usage)
    
    if m > 0:
        # Create a mapping from usage value to normalized rank
        # For ties, use average rank
        # Simpler approach: use the value itself normalized, but apply quadratic
        # Actually, let's use the rank of each edge's usage count
        
        # Get unique usage values and their ranks
        # For efficiency, we can use the normalized usage directly with a quadratic
        # But the directive says rank-based, so let's do proper ranking
        
        # Create rank array: for each edge, what is its rank among all edges by usage?
        # Use scipy.stats.rankdata if available, otherwise manual
        try:
            from scipy.stats import rankdata
            ranks = rankdata(usage_values, method='average')
        except ImportError:
            # Manual ranking
            order = np.argsort(usage_values)
            ranks = np.empty(m, dtype=np.float64)
            # For ties, assign average rank
            i = 0
            while i < m:
                j = i
                while j < m - 1 and usage_values[order[j + 1]] == usage_values[order[i]]:
                    j += 1
                avg_rank = (i + j + 2) / 2.0  # 1-indexed average
                ranks[order[i:j+1]] = avg_rank
                i = j + 1
        
        # Normalize ranks to [0, 1]
        normalized_ranks = (ranks - 1.0) / max(m - 1, 1)
        
        # Apply convex (quadratic) penalty: (normalized_rank)^2
        # This creates a steep barrier for high-usage edges
        rank_penalty_flat = normalized_ranks ** 2
        
        # Reconstruct the full matrix from the flat array
        rank_penalty = np.zeros_like(edge_distance, dtype=np.float64)
        rank_penalty[triu_mask] = rank_penalty_flat
        rank_penalty = rank_penalty + rank_penalty.T
        # Diagonal is already 0, and off-diagonal is set correctly
    else:
        rank_penalty = np.zeros_like(edge_distance, dtype=np.float64)
    
    # Combine penalties
    # Scale relative to mean distance for stability
    mean_dist = np.mean(edge_distance)
    if mean_dist > 0:
        tour_weight = 0.4 * mean_dist
        rank_weight = 0.25 * mean_dist
    else:
        tour_weight = 0.4
        rank_weight = 0.25
    
    total_penalty = tour_weight * penalty + rank_weight * rank_penalty
    
    # 3) Additional: small linear usage penalty for smoothness
    max_usage = edge_n_used.max()
    if max_usage > 0:
        linear_usage = edge_n_used / max_usage
        linear_weight = 0.05 * mean_dist if mean_dist > 0 else 0.05
        total_penalty += linear_weight * linear_usage
    
    # Final penalized distance matrix
    penalized = edge_distance + total_penalty
    
    # Ensure finiteness
    large_val = np.max(edge_distance) * 10 + 1.0
    penalized = np.where(np.isfinite(penalized), penalized, large_val)
    
    # Enforce symmetry
    penalized = (penalized + penalized.T) / 2.0
    
    # Zero out diagonal
    np.fill_diagonal(penalized, 0.0)
    
    return penalized
# EVOLVE-BLOCK-END
