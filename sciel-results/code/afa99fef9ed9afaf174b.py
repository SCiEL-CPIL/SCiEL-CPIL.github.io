import numpy as np
import time

def heuristics(prizes, weights):
    start_time = time.time()
    deadline = start_time + 250.0  # Safety margin before 300s limit
    
    prizes = np.asarray(prizes, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    if prizes.ndim == 1:
        prizes = prizes.reshape(1, -1)
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
    
    n_constraints = weights.shape[0]
    n_items = weights.shape[1]
    
    # Item prize: sum across rows
    item_prizes = np.sum(prizes, axis=0)
    
    capacity = 25.0
    
    # Total weight per constraint
    total_w = np.sum(weights, axis=1)
    
    # Bindingness: how binding each constraint is
    utilization = total_w / capacity
    binding = np.exp(2.0 * utilization)
    binding_sum = np.sum(binding)
    if binding_sum > 0:
        binding = binding / binding_sum
    
    # Per-constraint ratio
    safe_w = np.where(weights > 1e-9, weights, 1e-9)
    ratios = item_prizes[np.newaxis, :] / safe_w  # (n_constraints, n_items)
    
    # Weighted geometric mean of ratios
    log_ratios = np.log(np.where(ratios > 1e-9, ratios, 1e-9))
    weighted_log = np.sum(binding[:, np.newaxis] * log_ratios, axis=0)
    geo_mean = np.exp(weighted_log)
    
    # Harmonic mean (bottleneck)
    inv_ratios = 1.0 / np.where(ratios > 1e-9, ratios, 1e-9)
    harm_mean = float(n_constraints) / np.sum(inv_ratios, axis=0)
    
    # Min ratio (strictest bottleneck)
    min_ratio = np.min(ratios, axis=0)
    
    # Competitive exclusion score: 
    # For each item j, estimate opportunity cost by looking at items that 
    # would compete for the same constraint resources.
    # An item is "competing" with item j on constraint c if selecting j
    # would make it harder to select that other item due to constraint c.
    
    # For each constraint c, compute for each item j the sum of prizes of 
    # other items i that have high weight on constraint c (i.e., compete with j)
    # weighted by how much they compete.
    
    # Competition score: for item j on constraint c, the total "pressure" from
    # other items that share constraint c resources.
    # We approximate: items with high weight on constraint c compete more.
    
    # For each constraint c, compute competition matrix:
    # comp_c[j] = sum over i != j of item_prizes[i] * weights[c, i] * weights[c, j] / (capacity)
    # This represents the expected prize lost due to competition on constraint c.
    
    # More efficient: 
    # For constraint c, let w_c = weights[c, :] (shape n_items)
    # comp_c[j] = sum_i (item_prizes[i] * w_c[i]) * w_c[j] / capacity - item_prizes[j] * w_c[j]^2 / capacity
    # = (w_c[j] / capacity) * (sum_i item_prizes[i] * w_c[i] - item_prizes[j] * w_c[j])
    
    # Let S_c = sum_i item_prizes[i] * w_c[i]
    # Then comp_c[j] = (w_c[j] / capacity) * (S_c - item_prizes[j] * w_c[j])
    
    # Total competition score for item j = sum over c of binding[c] * comp_c[j]
    
    # Compute S_c for each constraint
    S = np.sum(item_prizes[np.newaxis, :] * weights, axis=1)  # (n_constraints,)
    
    # comp_c[j] = (weights[c, j] / capacity) * (S[c] - item_prizes[j] * weights[c, j])
    # Shape: (n_constraints, n_items)
    comp = (weights / capacity) * (S[:, np.newaxis] - item_prizes[np.newaxis, :] * weights)
    
    # Weight by bindingness
    comp_weighted = binding[:, np.newaxis] * comp
    
    # Total competition score per item
    comp_score = np.sum(comp_weighted, axis=0)
    
    # Competitive exclusion heuristic: prize relative to competition
    # Higher prize and lower competition = more attractive
    # Use: item_prizes / (1 + comp_score) or item_prizes - alpha * comp_score
    
    # Soft version: prize / (1 + beta * comp_score)
    # Normalize comp_score first
    comp_norm = np.where(comp_score > 0, comp_score, 1e-9)
    comp_mean = np.mean(comp_norm)
    if comp_mean > 0:
        comp_norm = comp_norm / comp_mean
    
    # Competitive exclusion score
    ce_score = item_prizes / (1.0 + 2.0 * comp_norm)
    
    # Check time
    if time.time() > deadline:
        # Return a simple fallback
        h = np.where(item_prizes > 0, item_prizes, 1e-9)
        return h
    
    # Blend with existing scores
    def norm_pos(a):
        a = np.where(a > 0, a, 1e-9)
        m = np.mean(a)
        if m > 0:
            a = a / m
        return a
    
    # Blend: 0.3 geo_mean + 0.2 harm_mean + 0.2 min_ratio + 0.3 ce_score
    h = 0.3 * norm_pos(geo_mean) + 0.2 * norm_pos(harm_mean) + 0.2 * norm_pos(min_ratio) + 0.3 * norm_pos(ce_score)
    h = np.where(h > 0, h, 1e-9)
    
    return h
