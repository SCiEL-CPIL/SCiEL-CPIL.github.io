import numpy as np

def heuristics(prizes, weights):
    # prizes: (5, 100), weights: (5, 100)
    # Calculate total prize per item
    total_prizes = np.sum(prizes, axis=0)
    
    # Calculate the maximum weight per item across constraints
    # This represents the "bottleneck" resource consumption
    max_weights = np.max(weights, axis=0)
    
    # Heuristic: Prize per unit of bottleneck weight
    # Add small epsilon to avoid division by zero
    heuristic = total_prizes / (max_weights + 1e-9)
    
    return heuristic
