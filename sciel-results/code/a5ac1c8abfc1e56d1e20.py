"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute an edge-attractiveness matrix for ACO orienteering.
    
    Design:
    - Prize term with slight boost to emphasize high-value targets
    - Distance penalty with exponent 2.5 for sharp contrast
    - Origin proximity bonus: favor nodes close to node 0 so ants can return
    - All entries strictly positive
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    n = len(prizes)
    
    # Distance to origin for each node (column j = distance from 0 to j)
    d_to_origin = distance_matrix[0, :]
    
    # Prize attractiveness: boost high prizes
    # Use prize^0.5 to slightly compress the range but still favor high prizes
    prize_term = np.sqrt(prizes[None, :]) + 1e-9
    
    # Distance penalty: penalize long moves strongly
    # exponent 2.5 gives good contrast
    dist_penalty = 1.0 / (distance_matrix + 1e-9) ** 2.5
    
    # Origin proximity bonus: nodes close to origin are easier to return to
    # This is a bonus that's higher for nodes near the origin
    origin_bonus = 1.0 / (d_to_origin[None, :] + 1e-9) ** 1.5
    
    # Combine: prize * distance_penalty * origin_bonus
    heuristic = prize_term * dist_penalty * origin_bonus
    
    # Ensure all entries are strictly positive
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
