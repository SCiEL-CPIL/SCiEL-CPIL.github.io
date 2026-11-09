import numpy as np
def heuristics(prizes, distance_matrix, max_len):
    n = len(prizes)
    # d_j0 is the distance from j to 0
    d_j0 = distance_matrix[:, 0]
    # Effective distance for move i->j
    # We want to penalize moves that consume too much of the budget
    # Budget is max_len.
    # The cost of visiting j from i is d_ij + d_j0 (since we must return from j)
    # So the "total cost" of this leg is d_ij + d_j0.
    
    # Heuristic: Prize / (Effective Distance)
    # To avoid division by zero and handle scale:
    # Normalize effective distance?
    
    eff_dist = distance_matrix + d_j0[None, :]
    # Add epsilon to prevent div by zero
    eff_dist = eff_dist + 1e-9
    
    # Use a power to sharpen the preference for efficient moves
    # alpha = 1.0 is standard. Let's try 1.0 first.
    # Or maybe 1.5 to be more selective.
    
    eta = prizes[None, :] / eff_dist
    
    # Ensure positivity
    eta = np.maximum(eta, 1e-9)
    
    return eta
