"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    """
    Computes attractiveness scores for items in a Multiple-Constraint Knapsack Problem.
    
    Strategy: Rank-based Aggregation.
    Instead of using a multiplicative geometric mean of efficiencies, which can be
    sensitive to outliers and scale, we use a rank-based approach.
    
    1. For each constraint j, compute the efficiency ratio prize_i / weight_ij.
    2. Rank the items for each constraint (1 = best efficiency).
    3. Compute the average rank for each item across all constraints.
    4. The heuristic score is inversely proportional to the average rank.
    
    This method robustly identifies items that are "good" across all dimensions,
    penalizing items that are excellent in one constraint but poor in another,
    without being skewed by extreme ratio values.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin for 300s limit

    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    num_constraints, num_items = prizes.shape

    # Check for immediate deadline breach
    if time.time() > deadline:
        return np.ones(num_items)

    # Calculate efficiency ratios: prize / weight
    # Shape: (num_constraints, num_items)
    epsilon = 1e-9
    # Add epsilon to weights to avoid division by zero
    ratios = prizes / (weights + epsilon)
    
    # Rank items for each constraint
    # argsort gives indices that would sort the array. 
    # We want rank 1 for the highest ratio.
    # np.argsort(np.argsort(x)) gives 0-based ranks in ascending order.
    # So, rank = num_items - 1 - argsort(argsort(x)) gives 0-based rank descending?
    # Let's be precise.
    # If x = [10, 20, 30], argsort(x) = [0, 1, 2]. argsort(argsort(x)) = [0, 1, 2].
    # Rank 0 is smallest. We want Rank 1 to be best (largest).
    # So, rank = num_items - 1 - argsort(argsort(x)).
    # If x = [10, 20, 30], sorted is [10, 20, 30].
    # Index 0 (val 10) -> rank 2 (worst).
    # Index 2 (val 30) -> rank 0 (best).
    # Let's use 1-based ranks for clarity in averaging.
    
    # Sort ratios in descending order to get ranks directly?
    # np.argsort(-ratios, axis=1) gives indices of sorted descending.
    # Then we need to invert this to get the rank of each original item.
    
    # Efficient ranking in NumPy:
    # rank = argsort(argsort(x)) gives 0..n-1 for ascending.
    # For descending (1=best), we can do:
    # ranks = np.argsort(np.argsort(-ratios), axis=1)
    # This gives 0 for the largest, 1 for the second largest, etc.
    
    # Let's verify:
    # ratios = [[10, 20, 30]]
    # -ratios = [[-10, -20, -30]]
    # argsort(-ratios) = [2, 1, 0] (index of max is 2, min is 0)
    # argsort(argsort(-ratios)) = argsort([2, 1, 0]) = [2, 1, 0]
    # So item 0 (val 10) gets rank 2. Item 2 (val 30) gets rank 0.
    # This is correct. 0 is best.
    
    ranks = np.argsort(np.argsort(-ratios, axis=1), axis=1)
    
    # Average rank across constraints
    # Lower average rank is better.
    avg_ranks = np.mean(ranks, axis=0)
    
    # Heuristic: Inverse of (avg_rank + 1) to avoid division by zero and keep positive
    # We add 1 to make the best rank 1, second 2, etc.
    # score = 1 / (avg_rank + 1)
    
    # To make the scores more spread out or compressed, we could use a power, 
    # but 1/x is a standard choice.
    heuristic_scores = 1.0 / (avg_ranks + 1.0)
    
    # Ensure all values are positive and finite
    heuristic_scores = np.where(np.isfinite(heuristic_scores), heuristic_scores, 1e-9)
    heuristic_scores = np.where(heuristic_scores > 0, heuristic_scores, 1e-9)
    
    return heuristic_scores
# EVOLVE-BLOCK-END
