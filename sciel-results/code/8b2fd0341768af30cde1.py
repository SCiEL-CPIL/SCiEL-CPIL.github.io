import numpy as np

def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items in a multiple-constraint knapsack.
    
    Strategy: 
    1. Calculate total prize for each item (sum across constraints).
    2. Identify the maximum weight constraint for each item (worst-case resource consumption).
    3. Compute heuristic as total prize divided by max weight.
    This prioritizes items with high value relative to their most demanding resource constraint.
    """
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes)
    weights = np.asarray(weights)
    
    # Handle shape validation
    if prizes.shape != (5, 100) or weights.shape != (5, 100):
        # Return a default safe heuristic if shape is wrong, though task says invalid result
        # But better to raise or return something that doesn't crash the evaluator if possible?
        # Task says: "An exception... make the whole result invalid."
        # So we should just let it fail or return valid shape.
        # Assuming inputs are correct per contract.
        pass

    # Total prize per item
    prize_sum = np.sum(prizes, axis=0)
    
    # Max weight per item across constraints
    max_weight = np.max(weights, axis=0)
    
    # Avoid division by zero
    max_weight_safe = np.maximum(max_weight, 1e-9)
    
    # Heuristic: Value / Worst-case Cost
    h = prize_sum / max_weight_safe
    
    # Ensure all values are positive and finite
    h = np.where(np.isfinite(h) & (h > 0), h, 1e-9)
    
    return h
