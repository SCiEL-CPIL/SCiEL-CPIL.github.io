"""
Strategy: Greedy Simulation Heuristic.
Instead of using Lagrangian relaxation (which estimates dual variables), 
we simulate a greedy knapsack solver to identify the "winning" set of items.
Items in the simulated optimal set are given high heuristic scores, 
guiding the ACO ants directly towards a known good solution structure.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, weights):
    """
    Computes attractiveness scores using a Greedy Simulation approach.
    
    1. We calculate a 'value density' for each item (sum of prizes / sum of weights).
    2. We simulate a greedy knapsack fill: repeatedly pick the item with the 
       highest value density that still fits within all 5 constraints (capacity 25 each).
    3. Items selected in this greedy simulation are assigned a high score.
       Items not selected are assigned a low score.
    4. To add nuance, the score for selected items is scaled by their individual 
       value density, so better items within the selected set are even more attractive.
    """
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Fallback for wrong shapes (though contract says shape is fixed)
    if prizes.shape != (5, 100) or weights.shape != (5, 100):
        total_prizes = np.sum(prizes, axis=0)
        total_weights = np.sum(weights, axis=0)
        heuristic = total_prizes / (total_weights + 1e-9)
        return np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)

    # 1. Calculate Value Density for each item
    # Sum prizes and weights across the 5 constraints for each of the 100 items
    total_prizes_per_item = np.sum(prizes, axis=0)  # shape (100,)
    total_weights_per_item = np.sum(weights, axis=0)  # shape (100,)
    
    # Avoid division by zero
    total_weights_safe = np.where(total_weights_per_item > 0, total_weights_per_item, 1e-9)
    value_density = total_prizes_per_item / total_weights_safe  # shape (100,)
    
    # 2. Greedy Simulation
    # Constraints: 5 constraints, capacity 25 each
    capacity = np.full(5, 25.0)
    current_load = np.zeros(5)
    
    # Indices of items, sorted by value density (descending) for initial ordering, 
    # but we will re-select dynamically based on fit.
    # Actually, a standard greedy approach picks the best density item that fits.
    # We need to efficiently find the item with max density among those that fit.
    
    # Since N=100 is small, we can do a simple loop.
    selected = np.zeros(100, dtype=bool)
    
    # To make the simulation robust, we might try a few different tie-breaking or 
    # slight variations, but standard greedy is usually sufficient for a heuristic hint.
    
    # We iterate until no more items can be added
    # We keep a list of available items
    available_indices = list(range(100))
    
    # Optimization: Pre-compute weights and prizes as lists of lists for faster access in loop?
    # Numpy access is fine for 100 items.
    
    while available_indices:
        # Find the item with the highest value density among available items that fits
        best_idx = -1
        best_density = -1e9
        
        # We can speed this up by sorting available_indices by density once, 
        # but since items get removed, we just scan the remaining ones.
        # For N=100, 100*100 = 10k ops, which is trivial.
        
        for idx in available_indices:
            # Check if item idx fits
            # weights[:, idx] < capacity - current_load
            if np.all(weights[:, idx] <= (capacity - current_load) + 1e-9):
                if value_density[idx] > best_density:
                    best_density = value_density[idx]
                    best_idx = idx
        
        if best_idx == -1:
            # No more items fit
            break
            
        # Select best_idx
        selected[best_idx] = True
        current_load += weights[:, best_idx]
        available_indices.remove(best_idx)
        
        # Time guard
        if time.time() - start_time > 290:
            break

    # 3. Assign Heuristic Scores
    # Items in the selected set get a high base score, scaled by their density.
    # Items not in the set get a low score.
    
    # The ACO probability is proportional to pheromone * heuristic.
    # Pheromone starts at 1. So initially, probability ~ heuristic.
    # We want the selected items to be very likely to be picked.
    
    # Let's set the score for selected items to be their value density (normalized).
    # And non-selected items to a small epsilon.
    
    # However, if we set non-selected to 1e-9, the ants will ONLY pick selected items 
    # in the first iteration. This might be too rigid if the greedy solution isn't perfect 
    # or if constraints change slightly due to random selection order in ACO.
    
    # A softer approach: 
    # Score = base_score + alpha * (is_selected * value_density)
    # But the contract says "larger is more attractive".
    
    # Let's use a two-tier scoring:
    # Selected items: High score (e.g., 10.0 + value_density)
    # Non-selected items: Low score (e.g., 1.0)
    
    # Why 10.0? To ensure they are significantly more attractive than the rest, 
    # guiding the colony, but not so exclusive that it prevents exploration if the 
    # greedy set has a flaw.
    
    heuristic = np.full(100, 1.0)
    heuristic[selected] = 10.0 + value_density[selected]
    
    # Normalize? The ACO uses the raw values. 
    # If we have 10 items with score ~15 and 90 items with score 1,
    # the probability of picking a selected item is ~ (10*15)/(10*15 + 90*1) = 150/240 = 62%.
    # This seems reasonable for a strong guide.
    
    # Ensure positivity and finiteness
    heuristic = np.where(np.isfinite(heuristic) & (heuristic > 0), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
