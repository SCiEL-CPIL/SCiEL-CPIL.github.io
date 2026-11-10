"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, weights):
    t_start = time.time()
    
    # Ensure correct shapes and types
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    if prizes.shape != (5, 100) or weights.shape != (5, 100):
        # Fallback to safe heuristic if shape is wrong, though task guarantees shape
        return np.ones(100)
        
    # Avoid division by zero
    weights_safe = np.maximum(weights, 1e-9)
    prizes_safe = np.maximum(prizes, 1e-9)
    
    # 1. Compute base score: Soft-min of ratios
    # Ratio of prize to weight for each constraint
    ratios = prizes_safe / weights_safe
    
    # Soft-min via log-sum-exp: softmin(x) = -log(sum(exp(-x)))
    # We use a temperature parameter to control smoothness.
    # Lower temperature -> closer to true min.
    temp = 0.05
    neg_ratios = -ratios / temp
    # Subtract max for numerical stability
    max_neg = np.max(neg_ratios, axis=0, keepdims=True)
    exp_neg = np.exp(neg_ratios - max_neg)
    sum_exp = np.sum(exp_neg, axis=0)
    soft_min_log = -np.log(sum_exp) + max_neg[0]
    soft_min_ratio = np.exp(soft_min_log)
    
    # 2. Bottleneck Penalty: Square the soft-min ratio
    # This amplifies the difference between items that are good on all constraints
    # and those that are bad on at least one.
    bottleneck_score = soft_min_ratio ** 2
    
    # 3. Co-occurrence Penalty
    # Perform a greedy packing to identify conflicts.
    # Sort items by base score (bottleneck_score) descending
    order = np.argsort(-bottleneck_score)
    
    capacity = 25.0
    remaining_capacity = np.full(5, capacity)
    selected_items = []
    rejected_items = []
    
    # Greedy packing simulation
    for idx in order:
        w_item = weights[:, idx]
        if np.all(w_item <= remaining_capacity):
            remaining_capacity -= w_item
            selected_items.append(idx)
        else:
            rejected_items.append(idx)
    
    # Calculate penalty for rejected items based on correlation with selected items
    # We want to penalize rejected items that are "similar" (high weight correlation)
    # to the selected items, as they likely compete for the same resources but were
    # worse. This discourages ants from picking these "near-miss" conflicts.
    
    penalty = np.zeros(100)
    
    if len(selected_items) > 0 and len(rejected_items) > 0:
        # Get weight vectors for selected and rejected items
        # Shape: (5, N_selected) and (5, N_rejected)
        w_sel = weights[:, selected_items]
        w_rej = weights[:, rejected_items]
        
        # Normalize weight vectors for correlation
        # Center and scale
        mean_sel = np.mean(w_sel, axis=0, keepdims=True)
        w_sel_centered = w_sel - mean_sel
        std_sel = np.std(w_sel_centered, axis=0, keepdims=True)
        std_sel = np.maximum(std_sel, 1e-9)
        w_sel_norm = w_sel_centered / std_sel
        
        mean_rej = np.mean(w_rej, axis=0, keepdims=True)
        w_rej_centered = w_rej - mean_rej
        std_rej = np.std(w_rej_centered, axis=0, keepdims=True)
        std_rej = np.maximum(std_rej, 1e-9)
        w_rej_norm = w_rej_centered / std_rej
        
        # Compute correlation matrix: (N_rejected, N_selected)
        # corr[i, j] is correlation between rejected item i and selected item j
        corr_matrix = (w_rej_norm.T @ w_sel_norm) / 5.0  # 5 constraints
        
        # For each rejected item, find the max correlation with any selected item
        max_corr = np.max(corr_matrix, axis=1)
        
        # Map correlation to penalty
        # If correlation is high (> 0), it's a strong conflict.
        # We apply a penalty proportional to the square of the positive correlation.
        # This is a smooth surrogate for a hard conflict.
        positive_corr = np.maximum(max_corr, 0.0)
        penalty_values = positive_corr ** 2
        
        # Assign penalties to the rejected items
        for i, idx in enumerate(rejected_items):
            penalty[idx] = penalty_values[i]
    
    # Apply penalty to the base score
    # We multiply the base score by (1 - penalty_factor * penalty)
    # This reduces the attractiveness of conflicting items.
    # Clamp to ensure positive values.
    penalty_factor = 0.5
    adjusted_score = bottleneck_score * (1.0 - penalty_factor * penalty)
    adjusted_score = np.maximum(adjusted_score, 1e-9)
    
    # 4. Sharpening: Raise to power 3 to increase differentiation
    final_scores = adjusted_score ** 3
    
    # Ensure all values are positive and finite
    final_scores = np.where(np.isfinite(final_scores) & (final_scores > 0), final_scores, 1e-9)
    
    return final_scores
# EVOLVE-BLOCK-END
