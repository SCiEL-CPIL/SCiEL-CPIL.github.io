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
    # An item i fits if for all c in 0..4: sum_{j in S} w[j, c] + w[i, c] <= 25.
    
    # Heuristic design:
    # We want items that have high prize and low weight in ALL constraints.
    # The "bottleneck" constraint for an item is the one where it consumes the most
    # relative to capacity.
    # Since capacity is constant (25), the bottleneck is the max weight across constraints.
    # But a better measure is how much it "crowds out" other items.
    
    # Strategy from inspiration: Harmonic mean of per-constraint ratios combined with min-ratio bottleneck emphasis.
    # Ratio r_c = prize / weight_c.
    # If weight_c is 0, ratio is inf.
    
    # Let's compute ratios for each constraint.
    # w is (5, 100)
    # item_prizes is (100,)
    
    # Avoid division by zero
    safe_weights = np.where(weights > 1e-9, weights, 1e-9)
    ratios = item_prizes / safe_weights  # (5, 100)
    
    # For each item, we want a score that reflects its efficiency across all constraints.
    # A simple approach: min ratio (bottleneck efficiency).
    # A better approach: Harmonic mean of ratios.
    # Harmonic mean = n / sum(1/r_i) = n / sum(w_i / prize) = n * prize / sum(w_i).
    # This is equivalent to prize / (sum(weights) / n) = prize / mean(weight).
    # So Harmonic mean of ratios is just prize / mean(weight).
    
    # Let's try a combination:
    # 1. min_ratio = min(ratios, axis=0)  # Bottleneck
    # 2. geo_mean = exp(mean(log(ratios), axis=0)) # Geometric mean
    # 3. harm_mean = 5 / np.sum(1/ratios, axis=0) # Harmonic mean
    
    # The inspiration mentions "min-ratio bottleneck emphasis and geometric slack".
    # Let's try: h = min_ratio * geo_mean
    # Or h = min_ratio ** 1.5 * harm_mean
    
    # Let's analyze the current best (0.9943) which was p / max_w.
    # p / max_w is prize / max(weight).
    # This is close to min_ratio (since min(p/w) is dominated by the largest w if p is constant, but p varies).
    # Actually, min(p/w_c) = p / max(w_c). So the current heuristic IS the min-ratio.
    
    # The inspiration says: "Harmonic mean ... combined with min-ratio bottleneck emphasis and geometric slack, then power=1.5 amplification".
    # Let's try: h = (min_ratio ** 1.5) * (harm_mean ** 0.5)
    # Or simply amplify the min_ratio more.
    
    # Let's calculate min_ratio and harm_mean.
    min_ratio = np.min(ratios, axis=0)
    harm_mean = 5.0 / np.sum(1.0 / ratios, axis=0)
    
    # Geometric slack? Maybe the spread of weights?
    # Let's stick to the ratios.
    
    # Proposed heuristic:
    # h = (min_ratio ** 1.2) * (harm_mean ** 0.8)
    # This emphasizes the bottleneck (min_ratio) while rewarding overall efficiency (harm_mean).
    
    # Let's try a simpler improvement first:
    # The current heuristic is min_ratio.
    # Maybe we should penalize items that are "unbalanced" in weight?
    # If an item has very low weight in one constraint and very high in another,
    # it might be hard to pack.
    
    # Let's try: h = min_ratio * (1 / (1 + std(weights, axis=0)))
    # This favors items with consistent low weights.
    
    # However, the measured improvement in inspiration 1 was 0.86 -> 1.06 using:
    # "Harmonic mean of per-constraint ratios combined with min-ratio bottleneck emphasis and geometric slack, then power=1.5 amplification"
    
    # Let's implement:
    # h = (min_ratio ** 1.5) * (harm_mean ** 0.5)
    
    # Or maybe:
    # h = min_ratio * harm_mean
    # Since min_ratio <= harm_mean, this downweights items where the bottleneck is much worse than the average.
    
    # Let's try: h = min_ratio ** 2 * harm_mean
    # This strongly emphasizes the bottleneck.
    
    # Let's go with: h = (min_ratio ** 1.5) * (harm_mean ** 0.5)
    
    h = (min_ratio ** 1.5) * (harm_mean ** 0.5)
    
    # Ensure positive
    h = np.where(h > 0, h, 1e-9)
    
    # Normalize to mean 1
    mean_h = np.mean(h)
    if mean_h > 0:
        h = h / mean_h
        
    return h
# EVOLVE-BLOCK-END
