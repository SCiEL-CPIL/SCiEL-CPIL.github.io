"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, distance_matrix, max_len):
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    # Set diagonal to a large value to prevent self-loops
    d = distance_matrix.copy()
    np.fill_diagonal(d, 1e10)
    
    # Heuristic: strongly favor high prizes, penalize distance
    # prize^alpha / (distance + eps)^beta
    # alpha > 1 amplifies high prizes; beta > 1 penalizes distance more
    eps = 1e-6
    alpha = 3.0
    beta = 1.5
    
    prize_factor = np.power(prizes + eps, alpha)
    dist_factor = np.power(d + eps, beta)
    
    heuristic = prize_factor[None, :] / dist_factor
    
    # Ensure positive finite values
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
