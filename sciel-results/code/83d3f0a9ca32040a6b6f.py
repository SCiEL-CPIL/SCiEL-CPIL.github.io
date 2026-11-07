import numpy as np

def heuristics(prizes, weights):
    # prizes: (5, 100), weights: (5, 100)
    # Calculate total prize per item
    total_prizes = np.sum(prizes, axis=0)
    
    # Calculate total weight per constraint
    total_weights_per_constraint = np.sum(weights, axis=1)
    
    # Capacity is 25 for all constraints
    capacity = 25.0
    scarcity_factors = total_weights_per_constraint / capacity
    
    # Calculate effective weight per item
    # weights.T is (100, 5), scarcity_factors is (5,)
    # Result is (100,)
    effective_weights = np.dot(weights.T, scarcity_factors)
    
    # Heuristic: Prize / Effective Weight
    # Add epsilon to avoid division by zero
    eps = 1e-9
    heur = total_prizes / (effective_weights + eps)
    
    # Ensure values are positive as per task description
    heur = np.maximum(heur, 1e-9)
    
    return heur
