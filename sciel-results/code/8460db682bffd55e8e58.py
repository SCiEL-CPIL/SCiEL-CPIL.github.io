import numpy as np

def heuristics(prizes, weights):
    # prizes: (5, 100), weights: (5, 100)
    # Ensure positive values to avoid division by zero
    eps = 1e-9
    p = prizes + eps
    w = weights + eps
    
    # Calculate inverse densities (weights/prizes) per constraint
    # Shape: (5, 100)
    inv_densities = w / p
    
    # Sum across constraints for each item
    # Shape: (100,)
    sum_inv_densities = np.sum(inv_densities, axis=0)
    
    # Harmonic mean: 5 / sum(1/d_i)
    heur = 5.0 / sum_inv_densities
    
    # Ensure positive and finite
    heur = np.maximum(heur, 1e-9)
    heur = np.nan_to_num(heur, nan=1e-9, posinf=1e9, neginf=1e-9)
    
    return heur
