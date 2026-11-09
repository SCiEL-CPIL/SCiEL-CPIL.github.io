"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for each item using a multi-objective dominance filter.
    
    Args:
        prizes: array of shape (5, 100), uniform in [0, 1]
        weights: array of shape (5, 100), uniform in [0, 1]
    
    Returns:
        array of shape (100,) with one score per item
    """
    n_items = prizes.shape[1]
    n_constraints = prizes.shape[0]
    capacity = 25.0
    
    # Compute total prize for each item (sum across constraints? No, prizes is per constraint)
    # Actually, prizes[i, j] is the prize contribution of item j to constraint i? 
    # Wait, the task says "prizes and weights, of shape (5, 100)" and "total prize found by any ant"
    # The total prize of an ant's solution is the sum of prizes of selected items.
    # Since prizes has shape (5, 100), I think prizes[i, j] is the prize of item j for constraint i?
    # But that doesn't make sense for "total prize". Let me re-read.
    
    # "mean over instances of the highest total prize found by any ant"
    # "each ant adds items... until none fits"
    # The total prize is likely the sum of all prize entries for selected items.
    # So total_prize = sum over all constraints and selected items of prizes[c, item]
    
    # For the heuristic, we want to identify Pareto-optimal items.
    # An item is dominated if there exists another item with >= prize in all constraints and > prize in at least one,
    # while having <= weight in all constraints and < weight in at least one.
    
    # But we need a scalar score. Let's use a dominance-based approach:
    # 1. For each item, compute its total prize (sum across constraints) and total weight (sum across constraints)
    # 2. Find Pareto-optimal items in the (total_prize, total_weight) space
    # 3. Assign high scores to Pareto-optimal items, low scores to others
    
    total_prize = prizes.sum(axis=0)  # shape (100,)
    total_weight = weights.sum(axis=0)  # shape (100,)
    
    # Find Pareto-optimal items: item i is non-dominated if no other item j has
    # total_prize[j] >= total_prize[i] and total_weight[j] <= total_weight[i] with at least one strict inequality
    is_pareto = np.ones(n_items, dtype=bool)
    
    for i in range(n_items):
        if not is_pareto[i]:
            continue
        for j in range(n_items):
            if i == j:
                continue
            # Check if j dominates i
            if (total_prize[j] >= total_prize[i] and total_weight[j] <= total_weight[i] and
                (total_prize[j] > total_prize[i] or total_weight[j] < total_weight[i])):
                is_pareto[i] = False
                break
    
    # Assign scores: Pareto-optimal items get high scores, others get low
    # Use the ratio of prize to weight for Pareto items to differentiate them
    scores = np.ones(n_items) * 1e-9
    
    # For Pareto-optimal items, assign score based on their efficiency
    pareto_indices = np.where(is_pareto)[0]
    if len(pareto_indices) > 0:
        # Compute prize/weight ratio for Pareto items
        ratios = total_prize[pareto_indices] / (total_weight[pareto_indices] + 1e-9)
        # Normalize ratios to [1, 10] or similar
        if len(ratios) > 1:
            min_r = ratios.min()
            max_r = ratios.max()
            if max_r > min_r:
                normalized = (ratios - min_r) / (max_r - min_r)
                scores[pareto_indices] = 1.0 + 9.0 * normalized
            else:
                scores[pareto_indices] = 5.0
        else:
            scores[pareto_indices] = 5.0
    
    # Ensure all scores are positive
    scores = np.maximum(scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END
