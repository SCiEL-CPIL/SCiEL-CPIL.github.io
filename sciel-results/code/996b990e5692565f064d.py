"""
Strategy: Global Optimization of Heuristic Parameters via Fast ACO Simulation
Instead of a static heuristic, we search for the optimal parameters of a 
"potential field" heuristic: h_ij = (capacity - (d_i + d_j))^-alpha.
We use the time budget to run a fast, vectorized simulation of the actual 
ACO process to evaluate candidate parameters and select the best one.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Fixed deadline guard: 330s wall-clock limit, keep 30s safety margin
DEADLINE = time.time() + 300.0

def _simulate_aco_fast(demands, capacity, heuristic, n_ants=20, n_iters=30, seed=0):
    """
    Fast vectorized simulation of the ACO algorithm to estimate the score.
    Returns the fewest bins used by any ant in the best iteration.
    """
    n = len(demands)
    rng = np.random.RandomState(seed)
    
    # Pheromone matrix
    pheromone = np.ones((n, n), dtype=np.float64)
    
    # Precompute fits matrix: can item j follow item i?
    # Note: The task says "placing item j right after item i in the same bin".
    # The bin state is not just i, but the sum of items in the bin.
    # However, the heuristic is h[current, j]. The "current" is the last item placed.
    # The constraint is that the new item j must fit in the *current bin*.
    # The current bin's load depends on the history, not just the last item.
    # Wait, the task says: "probability proportional to pheromone[current, j] * heuristic[current, j]"
    # And "When none fits, it opens a new bin."
    # The "fits" condition depends on the total load of the current bin.
    # Since the heuristic is static and computed once, but the "fits" condition is dynamic,
    # we must simulate the actual bin loading.
    
    # To make this fast, we can vectorize the ant steps? 
    # It's hard to fully vectorize because each ant has a different bin load.
    # But we can speed up the inner loop by using numpy operations where possible.
    
    best_bins = np.inf
    
    for it in range(n_iters):
        if time.time() > DEADLINE - 10:
            break
            
        # Initialize ants
        # Each ant starts with an empty bin.
        # "places a random fitting item"
        
        # We will track:
        # bin_loads: (n_ants,) current load of the active bin
        # last_item: (n_ants,) index of the last item placed (for heuristic lookup)
        # bins_used: (n_ants,) count of bins opened
        # placed: (n_ants, n) boolean mask of placed items? No, that's memory heavy.
        # Instead, we can track available items per ant? 
        # A simpler approach for simulation: 
        # Since n=120, n_ants=20, n_iters=30.
        # Total steps ~ 20 * 30 * 120 = 72000 steps.
        # Python loop might be slow. Let's try to optimize the inner loop.
        
        bin_loads = np.zeros(n_ants, dtype=np.float64)
        last_items = np.full(n_ants, -1, dtype=np.int64)
        bins_used = np.ones(n_ants, dtype=np.int64)
        
        # To check "fits", we need to know which items are already placed in the current bin?
        # No, "fits" means demand[j] <= capacity - bin_load.
        # We also need to ensure we don't place the same item twice.
        # The task says "item-by-item matrix". Usually implies permutation or subset.
        # "places a random fitting item" -> from remaining items?
        # "repeatedly picks a fitting item" -> from remaining items?
        # Yes, standard bin packing: each item is used exactly once.
        
        # We need to track which items are used.
        # used_masks: (n_ants, n) boolean
        used_masks = np.zeros((n_ants, n), dtype=bool)
        
        # For each ant, we need to pick the first item randomly.
        # Then iteratively pick the next.
        
        for a in range(n_ants):
            # Pick first item
            # Random fitting item: any item fits in empty bin.
            first_idx = rng.randint(n)
            used_masks[a, first_idx] = True
            bin_loads[a] = demands[first_idx]
            last_items[a] = first_idx
            
            # Iteratively pick next items
            # Loop until no more items fit
            while True:
                # Calculate remaining capacity
                rem_cap = capacity - bin_loads[a]
                
                # Find fitting items
                # Fits if demand[j] <= rem_cap and not used
                fits_mask = (demands <= rem_cap) & (~used_masks[a])
                
                if not np.any(fits_mask):
                    # Open new bin
                    # "When none fits, it opens a new bin."
                    # Does it place an item in the new bin immediately?
                    # "Each ant opens a bin, places a random fitting item, then repeatedly picks..."
                    # This structure implies the "place random" is the start of the bin.
                    # So when a bin is full, we open a new one and place a random fitting item.
                    
                    # Pick a random item from remaining
                    remaining = np.where(~used_masks[a])[0]
                    if len(remaining) == 0:
                        break # All items placed
                    
                    # Pick random from remaining
                    # Note: All remaining items fit in a new empty bin (since max demand < capacity)
                    new_idx = remaining[rng.randint(len(remaining))]
                    used_masks[a, new_idx] = True
                    bin_loads[a] = demands[new_idx]
                    last_items[a] = new_idx
                    bins_used[a] += 1
                    
                    # Continue loop to try to fill this new bin
                    continue
                
                # Calculate probabilities
                # P(j) proportional to pheromone[last, j] * heuristic[last, j]
                # Only for fitting items
                last = last_items[a]
                probs = pheromone[last, :] * heuristic[last, :]
                probs[~fits_mask] = 0.0
                
                total_prob = np.sum(probs)
                if total_prob == 0:
                    # Should not happen if heuristic is positive for fits
                    break
                    
                probs /= total_prob
                next_idx = rng.choice(n, p=probs)
                
                used_masks[a, next_idx] = True
                bin_loads[a] += demands[next_idx]
                last_items[a] = next_idx
                
                # Check if next item fits? 
                # The loop condition at the top handles this for the *next* iteration.
                # But we just placed it. The while loop continues.
                # If the bin is now full, the next iteration's fits_mask will be empty (or only items that don't fit).
                # Wait, if bin is full, rem_cap = 0. No item has demand <= 0.
                # So fits_mask will be empty -> triggers new bin.
                # This logic holds.
        
        # Update pheromone
        # "After each iteration, pheromone is multiplied by 0.9 and every entry gains 1/(fewest bins of that iteration)."
        # "fewest bins of that iteration" -> min(bins_used)
        min_bins = np.min(bins_used)
        if min_bins < best_bins:
            best_bins = min_bins
            
        pheromone *= 0.9
        pheromone += 1.0 / min_bins
        
    return best_bins

def heuristics(demands, capacity):
    demands = np.array(demands, dtype=np.float64)
    n = len(demands)
    
    # Define a family of heuristics:
    # h_ij = (capacity - (d_i + d_j))^-alpha  if d_i + d_j <= capacity else 1e-9
    # We search for optimal alpha.
    
    # Precompute waste matrix
    pair_sum = demands[:, None] + demands[None, :]
    waste = capacity - pair_sum
    fits = pair_sum <= capacity
    
    # Candidate alphas
    # Small alpha: less sensitive to waste. Large alpha: strongly prefers low waste.
    alphas = np.linspace(0.1, 5.0, 50)
    
    best_score = np.inf
    best_alpha = 1.0
    
    # Time budget check
    start_time = time.time()
    time_limit = 250.0 # Leave 50s for safety and overhead
    
    for alpha in alphas:
        if time.time() - start_time > time_limit:
            break
            
        # Construct heuristic
        # Avoid log(0) or division by zero
        # If waste is 0, h is inf? No, task says "not positive become 1e-9".
        # If waste is 0, (0)^-alpha is inf. We should handle this.
        # Actually, if waste is 0, it's a perfect fit. We want high attractiveness.
        # Let's use (waste + epsilon)^-alpha to avoid inf, or just clip.
        # The task says "Heuristic entries that are not positive become 1e-9".
        # Inf is not positive? It is positive. But Inf in probs might cause issues.
        # Let's clip waste to a small positive number to keep values finite.
        safe_waste = np.where(fits, np.maximum(waste, 1e-6), 1.0)
        
        h = np.where(fits, safe_waste ** (-alpha), 1e-9)
        
        # Ensure finite
        h = np.where(np.isfinite(h), h, 1e-9)
        h = np.where(h > 0, h, 1e-9)
        
        # Simulate ACO
        # Use a fixed seed for simulation to make it comparable across alphas?
        # The task says "seeded per instance". The evaluator seeds the instance.
        # But inside our heuristic function, we don't get the seed.
        # However, the ACO simulation inside the evaluator will use its own seed.
        # Our simulation here is just a proxy. We should use a representative seed or average.
        # Let's use seed 42 for the proxy simulation.
        
        score = _simulate_aco_fast(demands, capacity, h, n_ants=20, n_iters=30, seed=42)
        
        if score < best_score:
            best_score = score
            best_alpha = alpha
            
    # Construct final heuristic with best_alpha
    safe_waste = np.where(fits, np.maximum(waste, 1e-6), 1.0)
    final_h = np.where(fits, safe_waste ** (-best_alpha), 1e-9)
    
    # Final safety checks
    final_h = np.where(np.isfinite(final_h), final_h, 1e-9)
    final_h = np.where(final_h > 0, final_h, 1e-9)
    
    return final_h
# EVOLVE-BLOCK-END
