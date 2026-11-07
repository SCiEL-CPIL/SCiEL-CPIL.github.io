import numpy as np

def heuristics(distance_matrix):
    """
    Computes a rank-based heuristic matrix for the TSP.
    
    The heuristic value for an edge (i, j) is inversely proportional to its rank
    among all edges, where rank 1 is the shortest edge. This provides a robust
    bias towards shorter edges without being overly sensitive to the specific
    distance values, unlike power-law inverses.
    """
    d = np.asarray(distance_matrix, dtype=float)
    
    # Ensure valid square matrix
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be a square matrix")
        
    n = d.shape[0]
    
    # Create a copy to avoid modifying input
    d_work = d.copy()
    
    # Set diagonal to infinity so they are ranked last (or ignored)
    np.fill_diagonal(d_work, np.inf)
    
    # Flatten the matrix for ranking
    # We only care about upper triangle or all off-diagonals?
    # The heuristic matrix is usually symmetric. 
    # Let's rank all off-diagonal elements.
    flat_d = d_work.flatten()
    
    # Argsort to get ranks. 
    # We want the smallest distance to have rank 1.
    # argsort returns indices that would sort the array.
    # So, sorted_indices[0] is the index of the min value.
    sorted_indices = np.argsort(flat_d)
    
    # Create a rank array
    # rank_array[original_index] = rank (1-based)
    rank_array = np.zeros_like(flat_d)
    # Assign ranks 1 to N
    rank_array[sorted_indices] = np.arange(1, len(flat_d) + 1)
    
    # Reshape back to matrix
    rank_matrix = rank_array.reshape(d_work.shape)
    
    # Heuristic is 1/rank
    # Diagonal is rank ~N, so 1/rank is small, which is fine (we don't travel to self)
    # But to be safe, set diagonal to 0 or 1e-9 later.
    h = 1.0 / rank_matrix
    
    # The diagonal has a large rank, so small h.
    # We should explicitly zero out the diagonal or set to 1e-9 as per task "non-positive become 1e-9"
    # Actually, the task says "Heuristic entries that are not positive become 1e-9".
    # 1/rank is always positive.
    
    # Normalize to mean 1 to match the scale of initial pheromone (1.0)
    # We should probably normalize over the off-diagonal elements only?
    # The seed and previous versions normalized over the whole matrix or all elements.
    # Let's normalize over all elements to be consistent with "same shape" and typical ACO implementations.
    mean_h = np.mean(h)
    if mean_h > 0:
        h = h / mean_h
    
    # Ensure positivity
    h = np.where(h > 0, h, 1e-9)
    
    return h
