import numpy as np

def heuristics(prizes, weights):
    # Calculate the maximum weight per item across all constraints
    # This identifies the "bottleneck" constraint for each item
    # Items with a high max weight are harder to fit because they 
    # consume a large fraction of at least one capacity.
    
    # Ensure shapes are correct
    # prizes: (5, 100)
    # weights: (5, 100)
    
    # Using max is more robust to constraint-specific bottlenecks than sum
    # Sum averages out the impact, while max highlights the worst-case consumption.
    max_weights = np.max(weights, axis=0)
    
    # Calculate heuristic: Prize / Max Weight
    # Add epsilon to avoid division by zero
    eps = 1e-9
    h = prizes.sum(axis=0) / (max_weights + eps)
    
    # Ensure positive values
    h = np.maximum(h, 1e-9)
    
    return h
