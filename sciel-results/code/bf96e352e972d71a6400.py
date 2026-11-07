import numpy as np

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Global additive penalty strategy for Guided Local Search.
    
    Unlike the current program which only penalizes edges in the current tour,
    this approach penalizes ALL edges based on their historical usage frequency.
    This creates a 'memory' effect where edges that have been frequently used
    in previous local optima become more expensive, encouraging the 2-opt
    search to avoid them and explore new tour structures.
    
    The penalty is scaled by the maximum usage count to ensure that as the
    search stagnates (usage counts grow), the penalty pressure increases,
    helping to escape deep local minima.
    """
    if edge_n_used.size == 0:
        return edge_distance.copy()
    
    n = edge_distance.shape[0]
    
    # Find the maximum usage count to scale the penalty
    max_used = np.max(edge_n_used)
    if max_used == 0:
        return edge_distance.copy()
    
    # We want the penalty to be significant but not overwhelming.
    # A good baseline is to make the most-used edge effectively "forbidden" 
    # by adding a penalty comparable to the average edge length.
    # Let's estimate the average edge length from the distance matrix.
    # Since it's a symmetric matrix with 0s on the diagonal, we take the mean of upper triangle.
    triu_indices = np.triu_indices(n, k=1)
    avg_edge_len = np.mean(edge_distance[triu_indices])
    
    # Scale factor: 
    # We want penalty for max_used edges to be around 0.5 * avg_edge_len.
    # This means an edge used 'max_used' times gets a penalty that makes it 50% more expensive on average.
    # This is a strong signal to avoid it.
    # penalty = alpha * (count / max_used) * avg_edge_len
    # If count == max_used, penalty = alpha * avg_edge_len.
    # Let's try alpha = 0.5.
    
    alpha = 0.5
    
    # Calculate penalty matrix
    # penalty_matrix[i, j] = alpha * (edge_n_used[i, j] / max_used) * avg_edge_len
    penalty_matrix = (alpha / max_used) * edge_n_used * avg_edge_len
    
    # Add penalty to the distance matrix
    modified_distance = edge_distance + penalty_matrix
    
    # Ensure symmetry (just in case, though edge_n_used is symmetric)
    # The addition is element-wise, so symmetry is preserved if inputs are symmetric.
    
    return modified_distance
