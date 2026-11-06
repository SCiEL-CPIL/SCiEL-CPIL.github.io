"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    import time
    import random

    start_time = time.time()
    deadline = 55.0  # Safety margin before 90s kill (evaluator measures 60s, but we want to be safe)

    # Denominators are 2,3,...,48 -> indices 0..46
    n = 47
    denominators = list(range(2, 49))
    weights = [Fraction(1, d) for d in denominators]
    
    # We want to maximize the sum of selected weights such that sum < 1.
    # Greedy strategy: Sort by weight descending (denominator ascending) or ascending?
    # To get close to 1, we usually want to fill the "gap" with smaller denominators (larger weights) 
    # but we must be careful not to overshoot.
    # A common heuristic for "knapsack" like problems where capacity is 1:
    # Start with empty set.
    # Try adding items in order of decreasing weight (smallest denominator first)?
    # 1/2 = 0.5
    # 1/3 = 0.33 -> sum 0.83
    # 1/4 = 0.25 -> sum 1.08 (Too high)
    # So we skip 4.
    # 1/5 = 0.2 -> sum 1.03 (Too high)
    # 1/6 = 0.166 -> sum 0.996 (Valid)
    # This greedy approach (smallest denom first) tends to get very close to 1 quickly.
    
    # Let's try: Sort indices by denominator ascending (largest weight first).
    # Iterate and add if sum + weight < 1.
    
    indices_sorted = sorted(range(n), key=lambda i: denominators[i])
    
    best_sum = Fraction(0)
    best_indices = []
    
    # 1. Greedy Pass
    current_sum = Fraction(0)
    current_indices = set()
    
    for i in indices_sorted:
        if current_sum + weights[i] < 1:
            current_sum += weights[i]
            current_indices.add(i)
            
    if current_sum > best_sum:
        best_sum = current_sum
        best_indices = list(current_indices)
        
    # 2. Local Search / Swap Improvement
    # Try to improve by swapping selected items with unselected ones.
    # Or simply trying to add unselected items if they fit (greedy might have missed some if order was bad, 
    # but with sorted order, it's pretty good. However, swapping a large selected item for a smaller 
    # unselected one might not help increase sum. We want to INCREASE sum.
    # So we want to swap a selected item with a LARGER weight (smaller denominator) unselected item?
    # Wait, if we swap a selected item (weight w1) with unselected (weight w2) where w2 > w1,
    # the new sum is sum - w1 + w2. This increases the sum.
    # But we must ensure new_sum < 1.
    
    # Actually, the greedy "smallest denom first" fills with the biggest chunks.
    # The remaining gap is small.
    # Let's try a more robust local search:
    # Iterate through all pairs (selected, unselected).
    # If unselected weight > selected weight, and sum - selected + unselected < 1, do it.
    # This increases the sum.
    
    improved = True
    while improved and (time.time() - start_time) < deadline - 5:
        improved = False
        current_indices_set = set(best_indices)
        current_sum_frac = best_sum
        
        # Try swapping
        for i in list(current_indices_set):
            for j in range(n):
                if j in current_indices_set:
                    continue
                # Check if swapping i with j improves sum
                # New sum = current_sum - w[i] + w[j]
                # We want w[j] > w[i] to increase sum.
                if weights[j] <= weights[i]:
                    continue
                new_sum = current_sum_frac - weights[i] + weights[j]
                if new_sum < 1 and new_sum > current_sum_frac:
                    # Perform swap
                    current_indices_set.remove(i)
                    current_indices_set.add(j)
                    current_sum_frac = new_sum
                    improved = True
                    # Restart inner loop? Or just continue? 
                    # Restarting is safer to avoid missing chains, but slower.
                    # Given N=47, N^2 is small.
                    break
            if improved:
                # Update best and restart outer while? 
                # Or just update local variables and continue scanning?
                # Let's update local and continue scanning for more swaps in this pass.
                best_sum = current_sum_frac
                best_indices = list(current_indices_set)
                current_indices_set = set(best_indices)
        
    # 3. Randomized Hill Climbing (if time permits)
    # The deterministic swap might get stuck. Let's try some random moves.
    if (time.time() - start_time) < deadline - 10:
        random.seed(42)
        current_indices_set = set(best_indices)
        current_sum_frac = best_sum
        
        for _ in range(5000): # Limit iterations
            if (time.time() - start_time) > deadline - 5:
                break
            
            # Randomly pick a selected item and an unselected item
            if not current_indices_set:
                break
            i = random.choice(list(current_indices_set))
            # Pick j from unselected
            unselected = [x for x in range(n) if x not in current_indices_set]
            if not unselected:
                break
            j = random.choice(unselected)
            
            new_sum = current_sum_frac - weights[i] + weights[j]
            if new_sum < 1 and new_sum > current_sum_frac:
                current_indices_set.remove(i)
                current_indices_set.add(j)
                current_sum_frac = new_sum
                if current_sum_frac > best_sum:
                    best_sum = current_sum_frac
                    best_indices = list(current_indices_set)
                    
            # Also try just adding an unselected item if it fits (in case we have room)
            # The greedy phase should have maximized this, but swaps might have created room?
            # No, swaps keep cardinality same.
            # We could try adding an item if sum + w[j] < 1.
            # But greedy started with smallest denoms, so it's hard to add a larger denom (smaller weight) 
            # if we already have the small ones. 
            # However, if we swapped OUT a small denom (large weight) for a large denom (small weight), 
            # we might have room to add another large weight? 
            # Let's just stick to swaps for simplicity and speed.
            
    # Final Verification
    assert len(best_indices) > 0, "Must select at least one item"
    assert len(set(best_indices)) == len(best_indices), "Indices must be distinct"
    total_check = sum(weights[i] for i in best_indices)
    assert total_check < 1, f"Sum {total_check} not strictly below 1"
    
    return best_indices
# EVOLVE-BLOCK-END
