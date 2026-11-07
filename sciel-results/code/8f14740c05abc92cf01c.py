import numpy as np

def heuristics(prizes, weights):
    """
    Computes heuristic scores for items in a Multiple-Constraint Knapsack Problem.
    
    This heuristic uses a "density" approach where the prize is the total sum of prizes
    across all constraints, and the cost is the sum of weights normalized by the mean
    weight of each constraint. This transforms the multi-constraint problem into a
    single-dimensional density ratio, which is robust for uniform random instances.
    
    Args:
        prizes: numpy array of shape (5, 100), uniform in [0, 1].
        weights: numpy array of shape (5, 100), uniform in [0, 1].
        
    Returns:
        numpy array of shape (100,), where each entry is the attractiveness score.
    """
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # 1. Calculate total prize for each item (sum across constraints)
    total_prize = np.sum(prizes, axis=0)
    
    # 2. Calculate the mean weight for each constraint to normalize
    # This makes the weights comparable across different constraints
    mean_weight = np.mean(weights, axis=1)
    
    # Avoid division by zero
    mean_weight = np.where(mean_weight == 0, 1.0, mean_weight)
    
    # 3. Normalize weights per constraint
    # Shape: (5, 100)
    normalized_weights = weights / mean_weight[:, np.newaxis]
    
    # 4. Calculate total normalized cost for each item
    # Summing the normalized weights gives a scalar "total resource burden"
    total_cost = np.sum(normalized_weights, axis=0)
    
    # 5. Compute heuristic as prize density
    # Higher prize and lower total cost result in higher heuristic
    epsilon = 1e-9
    heuristic = total_prize / (total_cost + epsilon)
    
    # Ensure the output is a 1D array of floats
    return heuristic
