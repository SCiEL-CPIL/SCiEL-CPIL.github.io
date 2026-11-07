"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    # Use a small exponent on prize to mildly emphasize high-value nodes
    # and a slightly larger epsilon on distance to prevent extreme values
    # from very-close node pairs dominating the heuristic.
    prize_term = prizes + 1e-9
    dist_term = distance_matrix + 1e-4
    
    heuristic = prize_term[None, :] / dist_term
    
    # Ensure all values are positive and finite
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
