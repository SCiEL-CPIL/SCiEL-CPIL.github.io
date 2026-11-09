"""Guided local search penalty for TSP; Adaptive additive penalty with dynamic lambda."""
# EVOLVE-BLOCK-START
import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Applies an adaptive additive penalty to edges based on their historical usage count.
    Unlike multiplicative penalties, this adds a constant offset scaled by a dynamic lambda,
    which preserves the relative geometry of unused edges while pushing the search away
    from previously visited local optima. The penalty strength decays as the search
    explores more distinct edges, allowing for finer convergence in later stages.
    
    Args:
    edge_distance: np.ndarray, true distance matrix.
    local_opt_tour: list or np.ndarray, the current tour as a sequence of node indices.
    edge_n_used: np.ndarray, symmetric matrix of usage counts.
    
    Returns:
    np.ndarray: Modified distance matrix for the next 2-opt phase.
    """
    # Ensure inputs are numpy arrays for efficient operations
    D = np.asarray(edge_distance, dtype=float)
    N = np.asarray(edge_n_used, dtype=float)
    
    # Symmetrize inputs just in case
    D = (D + D.T) / 2.0
    N = (N + N.T) / 2.0
    
    # Calculate a dynamic penalty strength (lambda).
    # We want to start with a stronger penalty to escape the initial local optimum
    # and weaken it as the search matures.
    # Estimate "maturity" by the number of distinct edges used.
    # A fresh start has ~0 distinct edges. After 20 rounds, many edges will have been used.
    # For n=100, total possible edges ~ 4950.
    # distinct_edges = number of elements in N that are > 0.
    distinct_edges = np.sum(N > 0)
    
    # Base lambda. 
    # If distinct_edges is small, lambda should be larger.
    # If distinct_edges is large, lambda should be smaller.
    # Let's map distinct_edges to a lambda range.
    # Max distinct edges likely won't reach 4950 in 20 rounds of 2-opt, 
    # but could be a few hundred.
    # Let's use an exponential decay or inverse relationship.
    # lambda = base_lambda / (1 + distinct_edges / scale)
    
    base_lambda = 0.5  # Initial penalty strength
    scale = 50.0       # Controls how quickly lambda decays
    
    # As distinct edges increase, lambda decreases.
    # If distinct_edges = 0, lambda = 0.5.
    # If distinct_edges = 50, lambda = 0.25.
    # If distinct_edges = 100, lambda = 0.166.
    # This seems reasonable.
    
    lambda_val = base_lambda / (1.0 + distinct_edges / scale)
    
    # Apply additive penalty
    # new_dist = D + lambda * N
    # We must ensure the diagonal is 0 (distance to self)
    # N diagonal is likely 0, but let's be safe.
    
    new_dist = D + lambda_val * N
    
    # Ensure symmetry
    new_dist = (new_dist + new_dist.T) / 2.0
    
    # Ensure diagonal is 0
    np.fill_diagonal(new_dist, 0.0)
    
    # Safety check for finite values
    if not np.all(np.isfinite(new_dist)):
        return D.copy()
        
    return new_dist
# EVOLVE-BLOCK-END
