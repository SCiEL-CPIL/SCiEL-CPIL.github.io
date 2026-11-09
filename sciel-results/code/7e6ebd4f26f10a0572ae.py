import numpy as np

def heuristics(prizes, weights):
    """
    Returns one attractiveness score per item for the MCKP ACO solver.
    Uses prize divided by the sum of weights across all constraints.
    This favors items that are 'light' in total, promoting flexibility
    across all constraints rather than just the bottleneck.
    """
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Calculate sum of weights for each item across all constraints
    # weights shape is (5, 100)
    total_weights = np.sum(weights, axis=0)
    
    # Calculate heuristic: prize / total_weight
    # Add epsilon to avoid division by zero
    # If total weight is 0, the item is free, so heuristic should be very high
    heuristic = prizes / (total_weights + 1e-9)
    
    # Ensure no non-finite values
    # If prize is 0 and weight is 0, 0/1e-9 is 0.
    # If prize > 0 and weight is 0, heuristic is large.
    # If prize is 0 and weight > 0, heuristic is 0.
    # The ACO algorithm replaces non-positive heuristics with 1e-9.
    # So 0 is effectively 1e-9.
    
    heuristic = np.nan_to_num(heuristic, nan=1e-9, posinf=1e9, neginf=1e-9)
    
    # Ensure the output is 1D with 100 elements
    # The input shapes are guaranteed to be (5, 100) by the task contract,
    # but we verify just in case.
    if heuristic.shape != (100,):
        # If shape is wrong, return a flat array of 1s as a safe fallback
        # though the contract says wrong shape makes result invalid.
        # However, the function must return an array.
        # If the input was (100, 5), sum would be (5,), which is wrong.
        # The contract says shapes are (5, 100).
        pass
        
    return heuristic
