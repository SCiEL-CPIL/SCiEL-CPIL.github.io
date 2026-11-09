"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _ffd_assignment(demands, capacity):
    """
    Perform First-Fit-Decreasing bin packing.
    Returns a list where each element is the bin index for the corresponding item.
    """
    n = len(demands)
    # Sort items by size descending, keep track of original indices
    sorted_indices = np.argsort(-demands)
    
    bins = []  # list of (total_weight, [item_indices])
    assignment = [-1] * n
    
    for idx in sorted_indices:
        size = demands[idx]
        placed = False
        for b_idx in range(len(bins)):
            if bins[b_idx][0] + size <= capacity:
                bins[b_idx][0] += size
                bins[b_idx][1].append(idx)
                assignment[idx] = b_idx
                placed = True
                break
        if not placed:
            bins.append((size, [idx]))
            assignment[idx] = len(bins) - 1
            
    return assignment, bins

def _build_ffd_prior(demands, capacity):
    """
    Build a prior matrix based on FFD bin assignments.
    Items in the same bin get a high prior value.
    """
    n = len(demands)
    assignment, bins = _ffd_assignment(demands, capacity)
    
    # Initialize prior to 0
    prior = np.zeros((n, n))
    
    # For each bin, set prior[i][j] = 1 for all i, j in that bin
    for bin_items in bins:
        for i in bin_items:
            for j in bin_items:
                if i != j:
                    prior[i][j] = 1.0
                    
    return prior

def _fast_aco_eval(demands, capacity, heuristic, num_ants=20, num_iterations=30, seed=42):
    """
    Fast simulation of the ACO process to evaluate a heuristic.
    Returns the fewest bins used by any ant over all iterations.
    """
    rng = np.random.RandomState(seed)
    n = len(demands)
    d = np.array(demands, dtype=float)
    
    # Pheromone starts at 1
    pheromone = np.ones((n, n))
    
    best_bins_ever = float('inf')
    
    for iteration in range(num_iterations):
        best_bins_this_iter = float('inf')
        reward_sum = np.zeros((n, n))
        
        for ant in range(num_ants):
            # Ant opens a bin, places a random fitting item
            # Then repeatedly picks a fitting item with probability proportional to
            # pheromone[current, j] * heuristic[current, j]
            
            bins_used = 0
            current_bin_items = []
            current_bin_weight = 0.0
            
            while True:
                if not current_bin_items:
                    # Open a new bin, place a random fitting item
                    bins_used += 1
                    # Pick a random item that fits (any item fits in an empty bin)
                    # But we should only pick from remaining items
                    # For simplicity in this fast eval, we assume all items are available
                    # and we track used items per ant
                    pass
                
                # This is a simplification. The actual ACO tracks remaining items.
                # Let's implement a proper ant simulation.
                
                break
            
            # Proper Ant Simulation
            bins_used = 0
            used_items = np.zeros(n, dtype=bool)
            
            while not np.all(used_items):
                # Open new bin
                bins_used += 1
                current_bin_weight = 0.0
                last_item = -1
                
                # Place first item: random fitting item
                available = np.where(~used_items)[0]
                # All available items fit in an empty bin
                first_item = rng.choice(available)
                used_items[first_item] = True
                current_bin_weight += d[first_item]
                last_item = first_item
                
                # Repeatedly pick a fitting item
                while True:
                    # Find fitting items
                    # An item j fits if current_bin_weight + d[j] <= capacity and not used
                    fits = ~used_items
                    fits[d] = False  # placeholder, we need vectorized check
                    
                    # Vectorized check:
                    fits = ~used_items
                    fits = fits & (current_bin_weight + d <= capacity)
                    
                    if not np.any(fits):
                        break
                    
                    # Calculate probabilities
                    # prob[j] = pheromone[last_item, j] * heuristic[last_item, j]
                    probs = pheromone[last_item, :] * heuristic[last_item, :]
                    probs[~fits] = 0.0
                    
                    total_prob = np.sum(probs)
                    if total_prob == 0:
                        break
                        
                    probs = probs / total_prob
                    
                    # Pick next item
                    next_item = rng.choice(np.where(fits)[0], p=probs[fits])
                    used_items[next_item] = True
                    current_bin_weight += d[next_item]
                    last_item = next_item
                    
            # Record bins used
            if bins_used < best_bins_this_iter:
                best_bins_this_iter = bins_used
                
            # Accumulate reward for pheromone update
            # "every entry gains 1/(fewest bins of that iteration)"
            # This is done after the iteration, not per ant.
            # So we just need best_bins_this_iter.
            
        # After all ants in this iteration:
        if best_bins_this_iter < best_bins_ever:
            best_bins_ever = best_bins_this_iter
            
        # Pheromone update
        pheromone *= 0.9
        pheromone += 1.0 / best_bins_this_iter
        
    return best_bins_ever

def heuristics(demands, capacity):
    """
    Computes the heuristic matrix for the Ant Colony Optimization.
    Combines a structural prior from FFD with a local tightness heuristic.
    """
    start_time = time.time()
    
    d = np.array(demands, dtype=float)
    n = len(d)
    
    # 1. FFD Prior
    prior = _build_ffd_prior(d, capacity)
    
    # 2. Local Tightness Heuristic (1 / slack)
    pair_sums = d[:, None] + d[None, :]
    remaining = capacity - pair_sums
    
    # Initialize with 1e-9 for invalid
    h_tight = np.full((n, n), 1e-9)
    valid_mask = remaining >= 0
    eps = 1e-9
    valid_remaining = remaining[valid_mask] + eps
    h_tight[valid_mask] = 1.0 / valid_remaining
    
    # Normalize h_tight to be in [0, 1] or similar scale
    # Max value is 1/eps which is huge. Let's scale it.
    max_tight = np.max(h_tight)
    if max_tight > 0:
        h_tight = h_tight / max_tight
    
    # Combine: Heuristic = Prior * (1 + h_tight)
    # This rewards FFD-compatible pairs strongly, and among those, rewards tight fits.
    # Pairs not in FFD bins get 0 * (1 + h_tight) = 0, which becomes 1e-9 later.
    # Wait, if Prior is 0, the heuristic is 0. The problem says "Heuristic entries that are not positive become 1e-9".
    # So this effectively hard-codes the FDD structure.
    
    h = prior * (1.0 + h_tight)
    
    # Ensure positive values
    h[h <= 0] = 1e-9
    
    # Optional: Local search to improve heuristic if time permits
    # For now, return this combined heuristic.
    # We could try to optimize the mix ratio or use a different combination.
    
    return h
# EVOLVE-BLOCK-END
