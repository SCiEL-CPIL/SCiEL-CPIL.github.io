"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, weights):
    # prizes shape (5, 100), weights shape (5, 100)
    # Each constraint has capacity 25.
    # Strategy: Multi-start greedy simulation to estimate marginal utility.
    # We run many fast greedy constructions with different random orderings or perturbations.
    # For each item, we count how often it is selected in a valid solution.
    # Items that are frequently part of high-value valid solutions get higher scores.
    
    start_time = time.time()
    deadline = start_time + 10.0  # 10 seconds budget for heuristic computation
    
    n_constraints, n_items = weights.shape
    capacity = 25.0
    
    # Pre-compute item prizes (sum across constraints? or max? 
    # The task says "highest total prize". Usually sum of item values.
    # Let's assume item value is sum of prizes[:, j]
    item_values = prizes.sum(axis=0)
    
    # Selection counts
    counts = np.zeros(n_items)
    # Also track total value of solutions to potentially weight counts by quality?
    # Or just simple frequency. Let's try frequency first, maybe weighted by solution quality.
    
    # We will run several greedy passes.
    # Each pass:
    # 1. Shuffle items (or use a randomized priority).
    # 2. Iterate through shuffled items.
    # 3. If item fits in current knapsack (all constraints), add it.
    # 4. Record that item was selected.
    
    # To make it more "smart" than pure random shuffle, we can bias the shuffle
    # or use a greedy criterion with noise.
    # Let's try:
    # Base score = item_values / weights.max(axis=0) (the old heuristic)
    # Then for each iteration, we sample an ordering based on a noisy version of this score.
    # Or simply: random shuffle, but prefer items with higher value density?
    # Actually, pure random shuffle greedy is a good baseline for "marginal utility" in a dense graph.
    # But we have 300s total, and ACO takes some time. 10s is safe.
    
    # Let's do 500-1000 greedy passes depending on time.
    # Each pass is O(N^2) if we check all items against current state?
    # No, if we shuffle indices and iterate, checking fit is O(N*C). Total O(N^2*C) per pass.
    # N=100, C=5. 100^2 * 5 = 50,000 ops per pass.
    # 1000 passes = 50,000,000 ops. This is fast in NumPy if vectorized, but the loop is Python.
    # We can vectorize the "fit check" for a batch? No, the knapsack state changes sequentially.
    # So we must loop items.
    
    # Optimization:
    # Instead of shuffling the whole list every time, we can use a "random priority" approach.
    # Or just shuffle.
    
    # Let's try a hybrid:
    # 1. Calculate a base attractiveness (value density).
    # 2. Run greedy simulations where we pick the next item from a candidate set
    #    using a probability proportional to (base_attractiveness * random_noise).
    #    This is basically a stochastic greedy.
    
    # Simpler: Random shuffle greedy.
    # Let's see how many passes we can do in 5 seconds.
    
    # Pre-allocate weights for speed
    w = weights  # (5, 100)
    v = item_values # (100,)
    
    # We will run passes until time limit or max passes
    max_passes = 2000
    passes_done = 0
    
    # To make it faster, we can pre-compute the max weight per item for a quick reject?
    # No, we need to check all 5 constraints.
    
    # Vectorized check for a single item against current remaining capacity:
    # remaining_cap = np.array([25.0]*5)
    # if np.all(w[:, item_idx] <= remaining_cap):
    
    # Let's run the loop
    for _ in range(max_passes):
        if time.time() > deadline:
            break
        
        # Shuffle item indices
        order = np.random.permutation(n_items)
        
        # Current remaining capacity
        rem_cap = np.full(n_constraints, capacity)
        
        # Iterate through shuffled items
        for idx in order:
            # Check if item fits
            # w[:, idx] is (5,)
            if np.all(w[:, idx] <= rem_cap):
                # Add item
                rem_cap -= w[:, idx]
                counts[idx] += 1
        
        passes_done += 1
    
    # If we didn't do any passes (shouldn't happen), fallback
    if passes_done == 0:
        return item_values / (weights.max(axis=0) + 1e-9)
    
    # The score is the frequency of selection.
    # We can normalize or just use raw counts.
    # Since ACO multiplies by pheromone, the relative scale matters less than the ranking.
    # However, if counts are 0 for some items, they become 1e-9.
    
    # To add a bit of "value" awareness, we can multiply counts by the item's value?
    # Or just use counts. Counts already reflect that high-value items are often picked
    # if the greedy is good? Wait, pure random shuffle greedy picks items based on order.
    # It doesn't know about value!
    # So "random shuffle greedy" will pick items that are "small" (fit easily) but not necessarily "valuable".
    # This is a problem. We need the greedy to be value-aware.
    
    # Revised Strategy for the simulation:
    # Use a greedy that picks the item with the highest (value / weight_cost) ratio among feasible items.
    # But that's deterministic. We need variation.
    # Let's use a "noisy greedy":
    # At each step, consider all feasible items.
    # Pick one with probability proportional to (item_value / item_max_weight) * noise.
    # This is expensive: O(N^2) per pass if we scan all feasible items each step.
    # N=100. Steps <= 100. 100*100 = 10k ops per pass. 1000 passes = 10M ops. Doable.
    
    # Let's restart the logic with this better simulation.
    
    start_time = time.time()
    deadline = start_time + 5.0 # Tighter budget for the better algorithm
    
    counts = np.zeros(n_items)
    max_passes = 500
    
    # Pre-compute base attractiveness for the noisy greedy
    # base_attr = v / (weights.max(axis=0) + 1e-9)
    base_attr = v / (weights.max(axis=0) + 1e-9)
    
    for _ in range(max_passes):
        if time.time() > deadline:
            break
            
        rem_cap = np.full(n_constraints, capacity)
        remaining_items = set(range(n_items))
        
        # We need to pick items one by one.
        # To avoid O(N) scan for feasible items every step (which is fine for N=100),
        # we can just loop through remaining_items.
        
        while remaining_items:
            # Find feasible items
            feasible = []
            feasible_weights = []
            feasible_attrs = []
            
            for idx in remaining_items:
                if np.all(w[:, idx] <= rem_cap):
                    feasible.append(idx)
                    feasible_weights.append(w[:, idx])
                    feasible_attrs.append(base_attr[idx])
            
            if not feasible:
                break
                
            # Convert to arrays for vectorized noise sampling
            # We want to pick one with prob prop to attr * noise
            # Noise: log-normal or uniform? Let's use exponential noise (Gumbel max trick equivalent)
            # Or just multiply by uniform(0.5, 1.5)
            
            attrs = np.array(feasible_attrs)
            noise = np.random.uniform(0.5, 1.5, size=len(feasible))
            scores = attrs * noise
            
            # Pick index with max score
            best_local_idx = np.argmax(scores)
            chosen_idx = feasible[best_local_idx]
            
            # Update state
            rem_cap -= w[:, chosen_idx]
            counts[chosen_idx] += 1
            remaining_items.remove(chosen_idx)
            
    # If no passes, fallback
    if max_passes == 0 or counts.sum() == 0:
         return np.maximum(base_attr, 1e-9)
    
    # Combine frequency with base attractiveness?
    # The counts reflect "how often this item is useful in a good greedy path".
    # Let's return the counts.
    # Or counts * base_attr to emphasize high value?
    # Let's try just counts first, as it's a direct measure of "fit" in good solutions.
    # Actually, if an item has huge value but is huge, it might be picked less often in random-noise greedy
    # than a small high-value item. But we want the ACO to find the best total prize.
    # The ACO will handle the value accumulation. The heuristic should guide towards "good combinations".
    # So counts (frequency of co-occurrence in good greedy solutions) is a strong signal.
    
    # Ensure positive
    heuristic = np.maximum(counts, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
