import numpy as np

def heuristics(prizes, weights):
    # weights shape: (5, 100)
    # prizes shape: (5, 100)
    
    # Identify the bottleneck weight for each item
    # The item consumes capacity in all 5 dimensions.
    # The most restrictive dimension for this item is the one with max weight.
    max_weights = np.max(weights, axis=0) # shape (100,)
    
    # Heuristic: Prize value scaled by inverse of bottleneck weight.
    # This favors items that are high value and "light" in their tightest constraint.
    epsilon = 1e-9
    return prizes / (max_weights + epsilon)
