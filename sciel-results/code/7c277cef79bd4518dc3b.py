"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, weights):
    """
    Compute one attractiveness score per item for ACO of multi-constraint knapsack.
    
    prizes: (5, 100) or (100,) uniform [0,1]
    weights: (5, 100) uniform [0,1]
    Returns: (100,) array, larger = more attractive.
    """
    # Ensure correct shapes and types
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    if prizes.ndim == 2:
        p = prizes
    else:
        p = prizes.reshape(1, -1)
    
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
    
    n_items = weights.shape[1]
    
    # Calculate per-item metrics
    # p is (1, n_items) or (5, n_items)
    # If p has shape (5, 100), we need a single prize per item for the heuristic
    # The contract says "prizes and weights, of shape (5, 100)". 
    # Wait, the task says "prizes and weights, of shape (5, 100)".
    # Usually in MCKP, prizes are per item (1, n) and weights are (m, n).
    # But here it says prizes is (5, 100). This is unusual.
    # Let's re-read: "prizes and weights, of shape (5, 100), are uniform in [0, 1]."
    # This implies prizes is also (5, 100).
    # However, the scoring is "highest total prize".
    # If prizes is (5, 100), which prize is the "total prize"?
    # Usually, the objective is to maximize sum of prizes. If there are 5 prize vectors,
    # maybe it's the sum across constraints? Or maybe the first row?
    # Let's look at the current program: `p = np.mean(prizes, axis=0)`.
    # This assumes the prize to maximize is the average of the 5 prize rows.
    # Or maybe the prize is just one value per item, and the (5,100) shape is a quirk.
    # Let's assume the "prize" of an item is the sum or mean of its entries in the prize matrix.
    # Given the current code uses mean, and it scored 0.99, let's stick with a similar interpretation
    # but maybe sum is better if we want to maximize total?
    # Actually, if prizes is (5, 100), and we pick a subset of items, the total prize is likely
    # the sum of the selected items' prizes. If each item has 5 prize values, maybe the total prize
    # is the sum of all 5 values for each selected item? Or just one of them?
    # Let's assume the "prize" of an item is the sum of its row values if it's (5,100),
    # or just the value if it's (100,).
    # Let's use the sum across the first axis if 2D, as that represents the total "value" density.
    # However, the current code uses mean. Mean and sum differ by a constant factor (5),
    # which doesn't change the relative order in the heuristic if we normalize.
    # So let's use the sum or mean, it shouldn't matter much for ranking.
    # Let's use the sum to be safe about "total prize".
    
    if p.shape[0] == 1:
        item_prizes = p[0]
    else:
        # Sum across the 5 dimensions
        item_prizes = np.sum(p, axis=0)
    
    # Weights are (5, 100). Capacity is 25 for each of the 5 constraints.
    # Strategy: Model item attractiveness as expected marginal utility gain using shadow prices.
    # Shadow price for constraint c is estimated by inverse of total weight sum for that constraint.
    # Score = prize - sum_c (shadow_price_c * weight_c)
    # This penalizes items that consume scarce resources disproportionately.
    
    # Calculate total weight per constraint to estimate scarcity
    total_weights = np.sum(weights, axis=1)  # (5,)
    
    # Shadow prices: inverse of total weight (higher price for scarcer resources)
    # Add small epsilon to avoid division by zero
    shadow_prices = 1.0 / (total_weights + 1e-9)  # (5,)
    
    # Calculate weighted resource cost for each item
    # cost_c = shadow_price_c * weight_c for each constraint c
    # total_cost = sum_c cost_c
    resource_costs = np.sum(weights * shadow_prices[:, np.newaxis], axis=0)  # (100,)
    
    # Marginal utility = prize - resource_cost
    # We want this to be positive and larger for better items
    # If it's negative, the item is not worth taking given current scarcity
    # But we need positive values for heuristic, so we shift or use a different formulation
    
    # Alternative: Use the ratio of prize to resource cost, or prize * (1 + margin)
    # Let's try: h = item_prizes * (1.0 / (resource_costs + 1e-9))
    # This is prize / (sum_c shadow_price_c * weight_c)
    # Which is prize / (sum_c weight_c / total_weight_c)
    
    # Let's also consider the bottleneck constraint (min ratio) as a secondary factor
    # But the directive says to use the shadow price approach.
    
    # Let's try a blend:
    # h = item_prizes / (resource_costs + 1e-9)
    
    # Ensure safe division
    safe_costs = np.where(resource_costs > 1e-9, resource_costs, 1e-9)
    h = item_prizes / safe_costs
    
    # Ensure positive
    h = np.where(h > 0, h, 1e-9)
    
    # Normalize to mean 1
    mean_h = np.mean(h)
    if mean_h > 0:
        h = h / mean_h
        
    return h
# EVOLVE-BLOCK-END
