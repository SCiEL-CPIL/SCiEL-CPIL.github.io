"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    n = len(prizes)
    eps = 1e-9
    
    # d[j, 0]: distance from each node j back to origin
    d_j0 = distance_matrix[:, 0]  # shape (n,)
    
    # Heuristic: prize^1.5 / (d[i, j] + d[j, 0])
    # Structural change: Using sum of distances in denominator instead of product.
    # This models the total cost of the move i->j and the subsequent return to 0.
    # Power on prize (1.5) increases the relative attractiveness of high-prize nodes.
    
    # Numerator: prize raised to 1.5
    numerator = np.power(prizes, 1.5)
    numerator = numerator + eps # Ensure positive
    
    # Denominator: distance from i to j + distance from j to 0
    # distance_matrix is (n, n). d_j0 is (n,).
    # We want result[i, j] = numerator[j] / (distance_matrix[i, j] + d_j0[j])
    
    # d_j0[None, :] has shape (1, n)
    # distance_matrix has shape (n, n)
    denominator = distance_matrix + d_j0[None, :]
    denominator = denominator + eps
    
    result = numerator[None, :] / denominator
    
    # Ensure all values are positive and finite
    result = np.where(np.isfinite(result) & (result > 0), result, eps)
    
    return result
# EVOLVE-BLOCK-END
