import numpy as np

def heuristics(prizes, weights):
    """
    Computes heuristic scores for items in a Multiple-Constraint Knapsack Problem.
    
    The heuristic is designed to prioritize items that offer high prize value
    relative to their worst-case resource consumption (bottleneck constraint),
    using instance-specific normalization to ensure robustness across different
    problem instances.
    
    Args:
        prizes: numpy array of shape (5, 100), uniform in [0, 1].
        weights: numpy array of shape (5, 100), uniform in [0, 1].
        
    Returns:
        numpy array of shape (100,), where each entry is the attractiveness score.
    """
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # 1. Calculate total prize for each item
    # Summing across constraints gives the total value of the item.
    total_prize = np.sum(prizes, axis=0)
    
    # 2. Normalize prizes
    # We normalize by the maximum total prize to get a relative value score in [0, 1].
    max_total_prize = np.max(total_prize)
    if max_total_prize == 0:
        max_total_prize = 1.0 # Avoid division by zero, though unlikely with uniform [0,1]
    normalized_prize = total_prize / max_total_prize
    
    # 3. Normalize weights per constraint
    # For each constraint k, find the maximum weight among all items.
    # This allows us to compare items relative to the "heaviest" item in that dimension.
    max_weight_per_constraint = np.max(weights, axis=1)
    
    # Avoid division by zero if a constraint has all zero weights (unlikely but safe)
    max_weight_per_constraint = np.where(max_weight_per_constraint == 0, 1.0, max_weight_per_constraint)
    
    # Normalize weights to [0, 1] per constraint
    # Shape: (5, 100)
    normalized_weights = weights / max_weight_per_constraint[:, np.newaxis]
    
    # 4. Determine the bottleneck normalized weight for each item
    # An item is constrained by the resource it consumes the most (relative to the max in that resource).
    # We take the max across constraints for each item.
    bottleneck_weight = np.max(normalized_weights, axis=0)
    
    # 5. Compute heuristic
    # Heuristic is the ratio of normalized prize to bottleneck weight.
    # This favors items that are valuable and not bottlenecked by any single constraint.
    epsilon = 1e-9
    heuristic = normalized_prize / (bottleneck_weight + epsilon)
    
    # Ensure the output is a 1D array of floats
    return heuristic
