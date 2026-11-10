"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, weights):
    """
    Compute attractiveness scores for items using a regret-based heuristic.
    
    For each item, we compute how much the total prize of a baseline greedy
    solution drops if that item is forced to be excluded. This gives a
    discrete combinatorial signal: items with high regret are important
    to the solution and should be strongly attracted to, while items
    with low regret can be safely excluded.
    
    Args:
        prizes: array of shape (5, 100) - prizes per constraint per item
        weights: array of shape (5, 100) - weights per constraint per item
    
    Returns:
        array of shape (100,) - attractiveness scores per item
    """
    t0 = time.time()
    DEADLINE = 295.0  # safety margin before 300s limit
    
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Ensure positive values
    prizes = np.maximum(prizes, 1e-9)
    weights = np.maximum(weights, 1e-9)
    
    n_constraints, n_items = prizes.shape
    capacity = 25.0
    
    # Helper: greedy knapsack solver with constraint capacities
    def greedy_solve(prize_matrix, weight_matrix, capacity, excluded_item=None):
        """Solve multiple-constraint knapsack greedily by prize/weight ratio."""
        # Compute ratio for each item (use average ratio across constraints)
        ratios = prize_matrix / weight_matrix
        avg_ratio = ratios.mean(axis=0)
        
        # Get item indices sorted by ratio descending
        if excluded_item is not None:
            order = np.argsort(-avg_ratio)
            order = order[order != excluded_item]
        else:
            order = np.argsort(-avg_ratio)
        
        remaining = np.full(n_constraints, capacity, dtype=np.float64)
        total_prize = 0.0
        selected = []
        
        for idx in order:
            # Check if item fits in all constraints
            can_fit = np.all(weight_matrix[:, idx] <= remaining)
            if can_fit:
                remaining -= weight_matrix[:, idx]
                total_prize += prize_matrix[:, idx].sum()
                selected.append(idx)
        
        return total_prize, selected
    
    # Compute baseline solution (no exclusions)
    base_prize, _ = greedy_solve(prizes, weights, capacity)
    
    # Compute regret for each item
    regrets = np.zeros(n_items, dtype=np.float64)
    for i in range(n_items):
        if time.time() - t0 > DEADLINE:
            break
        excluded_prize, _ = greedy_solve(prizes, weights, capacity, excluded_item=i)
        regrets[i] = base_prize - excluded_prize
    
    # Ensure positive heuristics (negative regrets become small positive)
    heur = np.maximum(regrets, 1e-9)
    
    # Apply a power transformation to amplify differences
    # Items with much higher regret get disproportionately higher scores
    heur = heur ** 1.5
    
    # Normalize to reasonable range
    max_h = heur.max()
    if max_h > 0:
        heur = heur / max_h * 10.0  # Scale so max is 10
    
    # Final safety: ensure all positive
    heur = np.maximum(heur, 1e-9)
    
    return heur
# EVOLVE-BLOCK-END
