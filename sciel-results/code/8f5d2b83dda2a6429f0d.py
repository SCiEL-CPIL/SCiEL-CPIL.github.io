import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Capacity is fixed at 100 per task description
    capacity = 100.0
    golden_ratio_target = 0.618 * capacity
    
    # Initialize scores to a very low number (or -inf) for invalid bins
    scores = np.full(len(bins), -np.inf)
    
    # Find valid bins (enough remaining capacity)
    valid_mask = bins >= item
    
    if not np.any(valid_mask):
        # If no bin fits, return all -inf (or handle error, but contract implies valid input or we return something that doesn't crash)
        # The evaluator might handle this, but typically we should return something.
        # If no bin fits, the item is dropped or error. 
        # Assuming the evaluator only calls this if a bin exists or handles the -inf.
        return scores

    # Calculate new remainders for valid bins
    new_remainders = bins[valid_mask] - item
    
    # Calculate distance to target
    # We want to minimize distance, so we negate it for "highest score"
    distances = np.abs(new_remainders - golden_ratio_target)
    
    # Assign scores
    scores[valid_mask] = -distances
    
    return scores
