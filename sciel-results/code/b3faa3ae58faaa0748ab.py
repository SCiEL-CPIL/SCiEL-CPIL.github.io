import numpy as np

def heuristics(prizes, weights):
    # prizes: (5, 100), weights: (5, 100)
    # Ensure positive values to avoid division by zero
    eps = 1e-9
    
    # Aggregate prizes to a single scalar per item
    # Assuming the 5 rows represent components of the prize or are identical.
    # Summing is a safe bet for "total prize".
    item_prizes = np.sum(prizes, axis=0) + eps
    
    # Max weight across constraints for each item
    # Shape: (100,)
    max_weights = np.max(weights, axis=0) + eps
    
    # Heuristic: Prize / Max Weight
    heur = item_prizes / max_weights
    
    # Ensure positive and finite
    heur = np.maximum(heur, 1e-9)
    heur = np.nan_to_num(heur, nan=1e-9, posinf=1e9, neginf=1e-9)
    
    return heur
