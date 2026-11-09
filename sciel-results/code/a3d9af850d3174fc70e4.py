"""
Optimized heuristics for ACO CVRP using Rank-Based Distance Discrimination.

Strategy:
- Instead of using the cardinal value of the distance (1/d^alpha), we use the 
  ordinal rank of the distance among all edges.
- Rationale: Cardinal distance heuristics can be overly sensitive to the specific 
  scale of the instance and may create "cliffs" in the probability landscape 
  where the nearest neighbor is overwhelmingly more likely than the second nearest.
  Rank-based heuristics provide a smoother, more uniform gradient of attractiveness,
  allowing the pheromone trail (which accumulates over iterations) to more 
  effectively guide the search towards globally good routes without being 
  prematurely trapped by local geometric proximity.
- Normalization: The mean off-diagonal heuristic value is scaled to 1.0 to 
  balance with the initial pheromone value of 1.0.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ant colony optimization.
    
    Returns:
        numpy.ndarray: Heuristic matrix of the same shape as distance_matrix.
    """
    # Safety deadline guard (though this function is very fast)
    deadline = time.time() + 290.0

    d = np.asarray(distance_matrix, dtype=np.float64)
    n = d.shape[0]

    if n == 0:
        return np.array([])

    if d.shape != (n, n):
        # Fallback for unexpected shapes, though contract guarantees (n,n)
        return 1.0 / (d + 1e-9)

    eps = 1e-9
    
    # 1. Compute Ranks
    # We want the closest edge to have the highest attractiveness.
    # Rank 1 = smallest distance.
    # np.argsort returns indices that would sort the array.
    # We use a stable sort to handle ties deterministically if any exist (though unlikely in continuous coords).
    
    # Flatten the matrix to rank all pairwise distances globally.
    # This ensures that the "closest" edge in the entire instance gets the highest weight,
    # the second closest gets the next, etc.
    
    flat_d = d.flatten()
    
    # Get the ranks. 
    # argsort of argsort gives the rank of each element.
    # ranks[i] = number of elements smaller than flat_d[i].
    # We add 1 so the smallest distance gets rank 1.
    ranks = np.argsort(np.argsort(flat_d, kind='stable')) + 1
    
    # Reshape back to matrix
    rank_matrix = ranks.reshape(n, n)
    
    # 2. Base Heuristic: Inverse Rank
    # eta_ij = 1 / rank_ij
    # This creates a matrix where the closest neighbors have high values (e.g., 1, 0.5, 0.33...)
    # and far neighbors have low values.
    eta = 1.0 / rank_matrix.astype(np.float64)
    
    # 3. Normalization
    # Scale so that the mean of off-diagonal elements is 1.0.
    # This balances the heuristic with the initial pheromone (1.0).
    mask = ~np.eye(n, dtype=bool)
    mean_eta = np.mean(eta[mask])
    
    if mean_eta > 0 and np.isfinite(mean_eta):
        eta = eta / mean_eta
    else:
        # Fallback to simple inverse distance if something went wrong
        eta = 1.0 / (d + eps)
        mask = ~np.eye(n, dtype=bool)
        mean_eta = np.mean(eta[mask])
        if mean_eta > 0 and np.isfinite(mean_eta):
            eta = eta / mean_eta

    # 4. Sanitize
    # Ensure no negative or non-finite values
    eta[~np.isfinite(eta)] = 1e-9
    eta[eta <= 0] = 1e-9
    
    return eta
# EVOLVE-BLOCK-END
