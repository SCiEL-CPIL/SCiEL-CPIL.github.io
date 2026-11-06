import time
from fractions import Fraction

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin for 60s limit
    
    # Denominators 2 to 48 inclusive
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Precompute reciprocals as Fractions for exact arithmetic
    reciprocals = [Fraction(1, d) for d in denominators]
    
    # Sort indices by value descending (largest fractions first)
    # This helps the greedy upper bound be tighter and the search find good solutions early
    sorted_indices = sorted(range(n), key=lambda i: reciprocals[i], reverse=True)
    
    best_sum = Fraction(0)
    best_set = set()
    
    # Initial Greedy Solution to set a good lower bound
    current_sum = Fraction(0)
    current_set = set()
    for i in sorted_indices:
        if current_sum + reciprocals[i] < 1:
            current_sum += reciprocals[i]
            current_set.add(i)
            
    if current_sum > best_sum:
        best_sum = current_sum
        best_set = set(current_set)
        
    # Branch and Bound
    # State: index in sorted_indices, current_sum (Fraction), current_set (set)
    # We want to maximize sum subject to sum < 1.
    # Upper bound: current_sum + sum of all remaining elements (greedy fill)
    
    def upper_bound(idx, current_sum):
        """
        Calculate the maximum possible sum we could achieve starting from idx,
        assuming we can take any subset of remaining items (relaxation).
        Actually, a tighter bound is: current_sum + sum of all remaining items 
        that fit in the remaining capacity. But simple sum of all remaining is a valid UB.
        Even tighter: Greedy fill from idx with remaining capacity.
        """
        remaining_capacity = 1 - current_sum
        if remaining_capacity <= 0:
            return current_sum
            
        ub_sum = current_sum
        for j in range(idx, n):
            i = sorted_indices[j]
            if ub_sum + reciprocals[i] < 1:
                ub_sum += reciprocals[i]
            else:
                # Since sorted descending, if this one doesn't fit, smaller ones might fit?
                # No, if sorted descending, smaller ones are smaller. 
                # If 1/5 doesn't fit, 1/6 might fit if 1/5 > remaining_capacity.
                # So we can't break. We must continue.
                # However, for a fast UB, sum of all remaining is valid but loose.
                # Let's just add all remaining for a valid (though loose) UB.
                # Actually, the greedy fill above IS the tightest possible UB for this sorted list?
                # No, greedy fill is not an upper bound on the optimal subset sum.
                # Example: cap 10, items 6, 5, 5. Greedy takes 6. Opt is 5+5=10.
                # So greedy sum is a LOWER bound on the max achievable? No.
                # Wait, we are maximizing sum < 1.
                # The absolute maximum sum we can get is sum of ALL remaining items.
                # If sum_all_remaining + current_sum >= 1, it's not a valid UB for the "<1" constraint directly.
                # We need the max sum < 1.
                # A valid UB is: min(1, current_sum + sum(remaining_reciprocals)).
                # But since we need strictly < 1, the UB is just the sum of all remaining if it's < 1, 
                # or 1 if it's >= 1 (since we can't reach 1, the best we can do is close to 1).
                # For pruning: if current_sum + sum(remaining) <= best_sum, prune.
                pass
                
        # Simpler and tighter UB:
        # Sum of all remaining reciprocals added to current_sum.
        # If this sum <= best_sum, we can prune.
        sum_remaining = sum(reciprocals[sorted_indices[j]] for j in range(idx, n))
        return current_sum + sum_remaining

    # To make UB tighter and faster, precompute suffix sums of reciprocals in sorted order
    suffix_sums = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + reciprocals[sorted_indices[i]]

    def dfs(idx, current_sum, current_set):
        nonlocal best_sum, best_set
        
        # Time check
        if time.time() - start_time > timeout:
            return
            
        # Pruning: Upper Bound
        # Max possible sum from here is current_sum + sum of all remaining
        ub = current_sum + suffix_sums[idx]
        if ub <= best_sum:
            return
            
        # If we've considered all items
        if idx == n:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = set(current_set)
            return
            
        # Option 1: Skip item idx
        dfs(idx + 1, current_sum, current_set)
        
        # Option 2: Take item idx
        i = sorted_indices[idx]
        new_sum = current_sum + reciprocals[i]
        
        if new_sum < 1:
            current_set.add(i)
            dfs(idx + 1, new_sum, current_set)
            current_set.remove(i)
        else:
            # If new_sum >= 1, we cannot take it.
            # But we might still have room for smaller items.
            # The "Skip" branch above handles the case where we don't take this specific item.
            # We don't need to do anything else here because the "Skip" branch explores all subsets 
            # that do NOT include this item. The "Take" branch is invalid.
            pass

    # Start DFS
    dfs(0, Fraction(0), set())
    
    # Final verification
    if not best_set:
        # Fallback to simple greedy if DFS failed (shouldn't happen)
        for i in range(n):
            if sum(reciprocals[j] for j in best_set) + reciprocals[i] < 1:
                best_set.add(i)
                
    return list(best_set)
