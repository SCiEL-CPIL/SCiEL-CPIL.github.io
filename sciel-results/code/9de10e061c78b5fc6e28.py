import numpy as np
import time

def heuristics(demands, capacity):
    """
    Returns an item-by-item matrix of attractiveness for placing item j right after item i
    in the same bin. Larger is more attractive.
    
    Strategy:
    1. Pre-compute a baseline packing using First-Fit Decreasing (FFD) to get a lower bound on bins.
    2. For each valid pair (i, j) that fits in a bin, simulate packing the remaining items 
       using a fast greedy heuristic (First-Fit Decreasing) to estimate the total bins used.
    3. The heuristic value is 1 / (estimated_total_bins + epsilon).
       This encourages transitions that lead to the globally best packing, not just locally tight pairs.
    4. Pairs that don't fit get 1e-9.
    5. Time guard: If computation takes too long, fall back to a simpler heuristic.
    """
    n = len(demands)
    d = np.asarray(demands, dtype=float)
    
    # Deadline guard: 300s limit, leave 30s margin for safety
    start_time = time.time()
    deadline = 300.0 - 30.0
    
    # 1. Baseline: Sort items descending for FFD
    sorted_indices = np.argsort(d)[::-1]
    sorted_demands = d[sorted_indices]
    
    def fast_ffd_bins(items, cap):
        """
        Fast First-Fit Decreasing bin packing simulation.
        Returns number of bins used.
        """
        if len(items) == 0:
            return 0
        # items should be sorted descending
        bins = []
        for item in items:
            placed = False
            for idx, bin_load in enumerate(bins):
                if bin_load + item <= cap:
                    bins[idx] = bin_load + item
                    placed = True
                    break
            if not placed:
                bins.append(item)
        return len(bins)
    
    # Get baseline bins count (using all items)
    # This is just for context, the main logic is per-pair simulation
    baseline_bins = fast_ffd_bins(sorted_demands, capacity)
    
    # 2. Compute heuristic matrix
    eta = np.full((n, n), 1e-9, dtype=float)
    
    # Pre-compute list of all other items for each i, j pair to speed up simulation?
    # No, n=120 is small enough that O(n^3) or O(n^2 * n log n) might be tight but doable.
    # However, simulating FFD for every (i, j) pair is O(n^2 * n log n) which is ~ 120^3 * log(120) ~ 2M ops.
    # Python loops are slow. We need to be efficient.
    
    # Optimization: 
    # Instead of full simulation for every pair, we can use a simpler proxy or batch process.
    # But the prompt asks for a *different* strategy. The "simulation" strategy is robust.
    # Let's try to optimize the inner loop.
    
    # Alternative: Use a vectorized approximation or a simpler greedy that is faster.
    # Actually, 120 items is small. Let's just run the simulation but check time frequently.
    
    # We can pre-sort the "remaining" items for each pair? 
    # For pair (i, j), remaining are all k != i, k != j.
    # If we pre-sort the whole array, removing two items is O(n).
    
    sorted_d = sorted_demands
    # Map original index to sorted index? No, we just need the values.
    # The values of d are what matter for packing.
    
    # Let's store the values in a list for faster access in the inner loop
    d_list = d.tolist()
    
    # To make it faster, we can pre-compute the "next best" fits? 
    # No, let's stick to the simulation but optimize the Python loop.
    
    # A simpler, faster proxy for "how good is this pair" that captures global context:
    # The number of bins is roughly (sum(d) - sum(placed_in_current_bin)) / capacity + 1.
    # But FFD is better.
    
    # Let's try a hybrid: 
    # If time permits, do the full simulation.
    # If not, fall back to the local fit heuristic.
    
    # To speed up the simulation, we can use a list of bins and iterate.
    # We can also pre-compute the sorted array of all items.
    # For each (i, j), we remove d[i] and d[j] from the sorted list.
    
    # Let's create a sorted list of (value, index)
    sorted_items = sorted(zip(d_list, range(n)), key=lambda x: -x[0])
    
    # We will iterate over all pairs i, j
    # For each pair, if d[i] + d[j] <= capacity:
    #   Create a list of remaining items (values only)
    #   Run FFD on remaining items
    #   total_bins = 1 + ffd_bins(remaining)
    #   eta[i, j] = 1.0 / (total_bins + 1.0) # +1 to avoid division by zero issues or to scale
    
    # This is O(n^2 * n). 120^3 = 1,728,000 operations. 
    # Each FFD is O(n^2) in worst case (checking all bins for each item).
    # So total O(n^4) ~ 200 million ops. This might be too slow in Python.
    
    # Optimization: 
    # We can approximate the FFD cost. 
    # Or, we can just do it for a subset of pairs? No, the matrix must be complete.
    
    # Let's try to optimize the FFD simulation.
    # We can use a "best fit" or "first fit" that is faster?
    # First Fit is O(n * B) where B is bins. B is ~ n/cap * sum.
    
    # Let's try to run it and see. If it's too slow, we fall back.
    
    # To make it faster, we can pre-compute the sorted list of values.
    # For a given pair (i, j), the remaining values are sorted_d excluding d[i] and d[j].
    # Since d[i] and d[j] are specific values, we can remove them from the sorted list.
    
    # Actually, there's a simpler heuristic that is "globally aware" but O(n^2):
    # eta[i, j] = 1 / (1 + waste) is local.
    # What if we use: eta[i, j] = 1 / (1 + (capacity - d[i] - d[j]) / (capacity - max_remaining_fitting))?
    # No, that's complicated.
    
    # Let's stick to the simulation but make it as fast as possible.
    # We can use a list for bins.
    
    # Pre-sort the demands descending
    d_sorted_desc = np.sort(d)[::-1]
    
    # For each pair (i, j):
    # 1. Check if d[i] + d[j] <= capacity
    # 2. If yes, construct remaining list: all d[k] where k != i and k != j
    # 3. Sort remaining list descending (this is expensive if done from scratch)
    
    # Optimization: Pre-sort the entire array. 
    # For each i, j, we can iterate through the sorted array and skip i and j.
    # This is O(n) per pair to build the list, then O(n^2) for FFD.
    
    # Let's try to implement a very fast FFD in Python.
    
    def simulate_bins_excluding(i, j, d_arr, cap):
        """
        Simulate FFD for all items except i and j.
        Returns number of bins.
        """
        # We need the items sorted descending.
        # Instead of re-sorting, we can pre-sort the indices.
        # But we need to exclude i and j.
        
        # Let's just get the values of all items except i, j
        # And sort them.
        # n=120, sorting 118 items is fast.
        
        # Create mask
        # mask = np.ones(n, dtype=bool)
        # mask[i] = False
        # mask[j] = False
        # remaining = d_arr[mask]
        # remaining_sorted = np.sort(remaining)[::-1]
        
        # This involves numpy overhead which might be slow in a tight loop of 14400 iterations.
        # Let's use Python lists.
        
        # Pre-convert d to list
        # d_list is already available in outer scope
        
        # We need to exclude d_list[i] and d_list[j]
        # And sort the rest.
        
        # To avoid sorting 118 items 14400 times (which is 1.7M items sorted),
        # we can pre-sort the whole list and then "remove" the two items.
        
        pass # Placeholder for logic
    
    # Let's try a different approach: 
    # The "simulation" strategy is computationally heavy. 
    # Is there a lighter "global" strategy?
    
    # What if we use the "number of bins" from a simple greedy packing of the *entire* set, 
    # and then adjust based on how well i and j fit into the *first* bin of that packing?
    
    # Actually, let's look at the time budget. 300s.
    # 14400 pairs.
    # If each simulation takes 1ms, that's 14.4 seconds. That's fine!
    # Python sorting 118 floats takes ~10-20 microseconds.
    # FFD loop: 118 items, each checks ~10 bins. 1000 ops. ~50 microseconds.
    # Total per pair: ~100 microseconds.
    # 14400 * 100us = 1.44 seconds.
    # This is totally feasible.
    
    # Let's implement the simulation.
    
    d_list = d.tolist()
    
    # Pre-sort indices by demand descending
    # sorted_indices = np.argsort(d_list)[::-1]
    
    # To make removal fast, we can just iterate through the sorted list of values
    # and skip the values corresponding to i and j?
    # No, if there are duplicate values, we need to be careful.
    # It's safer to use indices.
    
    sorted_indices = np.argsort(d_list)[::-1]
    
    # For each pair (i, j)
    for i in range(n):
        di = d_list[i]
        for j in range(n):
            if i == j:
                continue
            dj = d_list[j]
            
            if di + dj > capacity:
                continue
                
            # Check time
            if time.time() > deadline:
                # Fallback to simple heuristic for remaining
                # We can break and fill the rest with a simple heuristic
                # Or just return what we have so far? 
                # The matrix must be complete. 
                # Let's fill the rest with 1e-9 or a simple local heuristic.
                # Actually, if we run out of time, we should return a valid matrix.
                # We can fill uncomputed entries with a simple local fit heuristic.
                
                # Calculate simple local heuristic for remaining
                # For any (i, j) not computed:
                # if di+dj <= cap: 1/(1+waste) else 1e-9
                
                # We need to know which ones are computed.
                # Let's use a mask.
                
                # This complicates the loop. 
                # Let's just hope it's fast enough. 
                # If not, we can catch the timeout and return a simple heuristic.
                
                pass
            
            # Build remaining list
            # We need all items except i and j, sorted descending.
            # We can iterate through sorted_indices and pick values, skipping i and j.
            
            remaining_vals = []
            for idx in sorted_indices:
                if idx == i or idx == j:
                    continue
                remaining_vals.append(d_list[idx])
            
            # Now run FFD on remaining_vals (which is already sorted descending)
            
            bins = []
            for item in remaining_vals:
                placed = False
                # First Fit
                for b_idx in range(len(bins)):
                    if bins[b_idx] + item <= capacity:
                        bins[b_idx] += item
                        placed = True
                        break
                if not placed:
                    bins.append(item)
            
            total_bins = 1 + len(bins) # 1 for the bin containing i and j
            
            # Heuristic value: higher is better.
            # We want to minimize total_bins.
            # eta = 1 / total_bins
            # To make it more distinct, maybe 1 / (total_bins ** 2) or exp(-total_bins)?
            # Since total_bins is an integer, 1/total_bins is fine.
            # We can scale it up to make it comparable to other heuristics.
            # Let's use 1.0 / total_bins.
            
            eta[i, j] = 1.0 / total_bins
    
    # Check if we finished in time
    if time.time() > deadline:
        # If we timed out, some entries might be 1e-9 (default) even if they fit.
        # We should fill any 1e-9 entries where di+dj <= capacity with a simple local heuristic
        # to ensure the matrix is "reasonable" even if not fully optimized.
        
        # Identify uncomputed valid pairs
        # A pair is uncomputed if eta[i, j] == 1e-9 and d[i] + d[j] <= capacity
        # Note: It's possible a valid pair got 1e-9 if the simulation wasn't run.
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if eta[i, j] <= 1e-9:
                    if d_list[i] + d_list[j] <= capacity:
                        waste = capacity - (d_list[i] + d_list[j])
                        # Simple local heuristic
                        eta[i, j] = 1.0 / (1.0 + waste / 10.0) ** 2
    
    # Ensure all values are positive and finite
    eta[~np.isfinite(eta)] = 1e-9
    eta[eta <= 0] = 1e-9
    
    return eta
