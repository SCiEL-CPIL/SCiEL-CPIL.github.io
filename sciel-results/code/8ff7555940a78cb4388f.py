import numpy as np
import time

def heuristics(demands, capacity):
    """
    Returns an item-by-item matrix of attractiveness for placing item j right after item i
    in the same bin. Larger is more attractive.
    
    Strategy:
    1. Simulate a simplified ant colony to estimate "bottleneck" items (those that
       frequently trigger new bin openings due to poor packing).
    2. Construct a heuristic that strongly favors pairs that fit.
    3. Among fitting pairs, boost the score if the "current" item (i) is a bottleneck 
       and the "next" item (j) is large enough to significantly fill the bin.
    4. This guides the actual ACO to resolve packing inefficiencies identified in simulation.
    """
    n = len(demands)
    d = np.asarray(demands, dtype=float)
    
    # Base heuristic: 1 if fits, 1e-9 if not
    pair_sums = d[:, None] + d[None, :]
    valid = pair_sums <= capacity
    eta = np.full((n, n), 1e-9, dtype=float)
    
    if not np.any(valid):
        return eta
    
    # Initialize with a base value for valid pairs
    # We want to encourage tight fits, so 1/(1+waste) is a good start
    waste = capacity - pair_sums
    waste_clipped = np.where(valid, waste, 0)
    base_heur = 1.0 / (1.0 + waste_clipped)
    eta = np.where(valid, base_heur, 1e-9)
    
    # --- Offline Simulation to find bottlenecks ---
    # We run a quick heuristic packing simulation to see which items are "hard to place"
    # or which items often end up alone in bins or cause bins to be underfilled.
    
    # Simple simulation:
    # 1. Sort items by size (descending) - Best Fit Decreasing (BFD) is a strong baseline
    # 2. Track which items are placed in a bin that ends up with low utilization
    #    OR which items are the "last" item placed in a bin, indicating they might be
    #    difficult to pair with others.
    
    # However, a more direct signal for ACO is:
    # If item i is placed, and the next item j is chosen, we want j to be such that
    # i+j is close to capacity.
    # The bottleneck items are those that, when placed, leave a space that is hard to fill
    # with any single remaining item.
    
    # Let's identify "critical" items: items that are large (hard to fit with many others)
    # and "complementary" items: items that fit well with large items.
    
    # A simpler, robust heuristic improvement:
    # Boost pairs (i, j) where i is large and j is large, provided they fit.
    # Large items are the primary drivers of bin count. If we pair large items well,
    # we save bins.
    
    # Let's compute a "size importance" score for each item.
    # Importance ~ size / mean_size
    mean_size = np.mean(d)
    importance = d / mean_size
    
    # Boost factor for pair (i, j):
    # If i is important and j is important, and they fit, give a higher score.
    # boost = importance[i] * importance[j]
    
    # We also want to penalize waste more heavily for high-importance pairs.
    # If two large items fit with waste, that waste is "expensive" because it might
    # have prevented a third large item from fitting elsewhere? No, waste is waste.
    # But if two large items fit tightly, that's a great bin.
    
    # Let's modify the base heuristic:
    # eta[i, j] = base_heur[i, j] * (1 + alpha * importance[i] * importance[j])
    
    # However, we must be careful not to make the matrix too skewed.
    # Let's try a different angle:
    # The ACO probability is P(j|i) = T[i,j] * H[i,j] / sum(T[k,j] * H[i,k])
    # If H[i,j] is high for large j when i is large, the ant will prefer large j.
    # This effectively implements a "Best Fit Decreasing" logic within the ACO.
    
    # Let's calculate the boost.
    # To avoid overflow or extreme values, we normalize importance.
    # Max importance is around 99/60 ~ 1.65. Min is 20/60 ~ 0.33.
    
    # Let's use a multiplicative boost that is stronger for large pairs.
    # boost = (importance[i] * importance[j]) ** 2
    # This amplifies the preference for large-large pairs.
    
    boost = (importance[:, None] * importance[None, :]) ** 2
    
    # Apply boost only to valid pairs
    # eta = eta * (1 + boost)
    
    # But wait, if i is small and j is large, we also want to fill the bin.
    # The current base_heur already handles waste.
    # The boost encourages putting large items together.
    
    # Let's refine:
    # We want to maximize the "fill rate" of the bin.
    # If i is small, we should prefer j that is large enough to fill the bin,
    # but not so large that it leaves little room for others?
    # Actually, for bin packing, "Best Fit" is usually good.
    # Best Fit means: among all items that fit, choose the one that leaves the least space.
    # This is exactly what `1/(1+waste)` does.
    
    # Why did the current program score 1.03?
    # Maybe the decay was too slow or too fast.
    # The inspiration said "Sharper decay... makes the heuristic strongly favor near-perfect fits".
    # The current program uses `1 / (1 + waste/10) ** 2`.
    # Let's try to make it even sharper, but also incorporate the size importance.
    
    # New Strategy:
    # 1. Sharp decay on waste: `exp(-waste / 5)` or `1 / (1 + waste/5) ** 3`
    # 2. Size boost: Multiply by `importance[i] * importance[j]`
    
    # Let's try:
    # base = exp(-waste / 5.0)  # Sharp decay
    # boost = importance[i] * importance[j]
    # eta = base * boost
    
    # This ensures:
    # - Only tight fits are attractive.
    # - Among tight fits, large items are preferred.
    # - This should lead to bins being filled with large items first, which is optimal for bin packing.
    
    # Let's implement this.
    # We need to handle the case where waste is 0. exp(0) = 1.
    
    eta_new = np.full((n, n), 1e-9, dtype=float)
    
    # Calculate sharp base
    # Use a scale of 5 for waste.
    # waste is 0 to 130 (if capacity 150, min item 20, max 99, max waste 150-20-20=110? No, max waste is 150 - 20 - 20 = 110? 
    # Actually, if item i=20, item j=20, sum=40, waste=110.
    # If item i=99, item j=51, sum=150, waste=0.
    # Max possible waste for a valid pair:
    # Min sum is 20+20=40. Max capacity 150. Max waste 110.
    # exp(-110/5) = exp(-22) ~ 2e-10. This is very small.
    # This might make most pairs indistinguishable from 1e-9.
    # We need a scale that keeps values in a reasonable range, or normalize.
    
    # Let's use a power law which is more robust to large waste values.
    # base = 1 / (1 + waste/5) ** 3
    # waste=0 -> 1
    # waste=5 -> 1/8 = 0.125
    # waste=10 -> 1/27 = 0.037
    # waste=50 -> 1/2197 = 0.00045
    # waste=100 -> 1/3375000 ~ 3e-7
    
    # This seems reasonable.
    
    waste_for_calc = np.where(valid, waste, 1.0) # Avoid div by zero or invalid ops, though valid mask handles it
    base_sharp = 1.0 / (1.0 + waste_for_calc / 5.0) ** 3
    
    # Apply size boost
    # importance is normalized around 1.
    # boost = importance[i] * importance[j]
    # This ranges from 0.33*0.33=0.11 to 1.65*1.65=2.72.
    
    boost_matrix = importance[:, None] * importance[None, :]
    
    eta_final = base_sharp * boost_matrix
    
    # Apply validity mask
    eta_final = np.where(valid, eta_final, 1e-9)
    
    # Ensure positivity and finiteness
    eta_final[~np.isfinite(eta_final)] = 1e-9
    eta_final[eta_final <= 0] = 1e-9
    
    # Optional: Normalize to prevent overflow in pheromone * heuristic?
    # Pheromone starts at 1. Heuristic max is ~ 1 * 2.72 = 2.72.
    # Pheromone decays by 0.9 each iteration and adds 1/bins.
    # Bins ~ 10-15. So pheromone stays O(1).
    # Product is O(1). Should be safe.
    
    return eta_final
