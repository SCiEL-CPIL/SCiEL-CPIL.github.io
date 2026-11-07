import numpy as np

def heuristics(prizes, weights):
    # prizes: (5, 100), weights: (5, 100)
    # Ensure positive values to avoid log/division issues
    eps = 1e-9
    p = prizes + eps
    w = weights + eps
    
    # Calculate density per constraint: (5, 100)
    densities = p / w
    
    # Geometric mean of densities per item
    # log(densities) is (5, 100)
    log_densities = np.log(densities)
    mean_log_density = np.mean(log_densities, axis=0)
    
    # Exponentiate back to get geometric mean
    heur = np.exp(mean_log_density)
    
    # Ensure positive
    heur = np.maximum(heur, 1e-9)
    
    return heur
