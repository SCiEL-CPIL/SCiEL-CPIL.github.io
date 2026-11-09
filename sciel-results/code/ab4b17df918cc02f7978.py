# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    denominators = list(range(2, 49))  # [2, 3, ..., 48]
    n = len(denominators)
    
    # Precompute reciprocals
    reciprocals = [Fraction(1, d) for d in denominators]
    
    # Greedy initial solution
    selected = []
    total = Fraction(0)
    for i in range(n):
        if total + reciprocals[i] < 1:
            selected.append(i)
            total += reciprocals[i]
    
    best_indices = selected[:]
    best_deficit = 1 - total  # This is the deficit (1 - sum)
    
    # Now try to improve with local search / branch and bound
    # Try adding/removing elements to reduce deficit
    
    # Branch and bound: explore subsets in order of potential to get close to 1
    # Use a priority queue approach: states are (current_sum, current_indices, next_start_index)
    # We want to find states where current_sum is close to 1 but < 1
    
    # Actually, let's use a smarter approach: 
    # The deficit is 1 - sum. We want to minimize this.
    # Equivalent: maximize sum subject to sum < 1.
    
    # Let's do a DFS with pruning, ordered by descending reciprocal value
    # to quickly build up sums close to 1
    
    # Sort by reciprocal value descending (i.e., denominator ascending)
    # But we need to track original indices
    
    # Let's try a different approach: meet in the middle
    # Split into two halves, compute all subset sums for each half,
    # then find pairs that sum to just below 1
    
    # But 2^23 is too large. Let's use a smarter search.
    
    # Alternative: use the fact that we only need to get close to 1.
    # Start with a good greedy solution, then try local improvements.
    
    # Local search: try swapping one element for another, or adding/removing
    improved = True
    while improved and (time.time() - start_time) < deadline - 5:
        improved = False
        current_set = set(best_indices)
        current_sum = sum(reciprocals[i] for i in best_indices)
        current_deficit = 1 - current_sum
        
        # Try removing one element and adding a different one
        for remove_idx in list(current_set):
            for add_idx in range(n):
                if add_idx in current_set:
                    continue
                new_sum = current_sum - reciprocals[remove_idx] + reciprocals[add_idx]
                if new_sum < 1:
                    new_deficit = 1 - new_sum
                    if new_deficit < current_deficit:
                        best_indices = [i for i in best_indices if i != remove_idx]
                        best_indices.append(add_idx)
                        best_deficit = new_deficit
                        improved = True
                        break
            if improved:
                break
        
        if improved:
            continue
            
        # Try adding one element (if current sum + new < 1)
        current_set = set(best_indices)
        current_sum = sum(reciprocals[i] for i in best_indices)
        for add_idx in range(n):
            if add_idx in current_set:
                continue
            new_sum = current_sum + reciprocals[add_idx]
            if new_sum < 1:
                new_deficit = 1 - new_sum
                if new_deficit < best_deficit:
                    best_indices.append(add_idx)
                    best_deficit = new_deficit
                    improved = True
                    break
        
        if improved:
            continue
            
        # Try removing one element (to make room for a better combination)
        # This might not help directly, but let's try
        for remove_idx in list(best_indices):
            new_sum = sum(reciprocals[i] for i in best_indices if i != remove_idx)
            if new_sum < 1:
                new_deficit = 1 - new_sum
                # This makes deficit worse, so only useful if we can then add something better
                # Skip for now
    
    # Try a more systematic search with time limit
    # Use a beam search / A* approach
    
    # Actually, let me try a different structural approach:
    # Work with the deficit directly. Deficit = 1 - sum = 1 - sum(1/d_i)
    # = (LCM - sum(LCM/d_i)) / LCM
    # But LCM is huge. Let's stick with Fraction.
    
    # Let me try a randomized local search with many restarts
    import random
    
    best_deficit_final = best_deficit
    best_indices_final = best_indices[:]
    
    # Try more aggressive local search: 2-swap moves
    improved = True
    while improved and (time.time() - start_time) < deadline - 5:
        improved = False
        current_set = set(best_indices_final)
        current_sum = sum(reciprocals[i] for i in best_indices_final)
        
        # Try 2-swap: remove two, add two different ones
        # This is O(n^4) which is 47^4 ~ 5 million, might be too slow
        # Let's do it more carefully
        
        # Try single add/remove first
        for add_idx in range(n):
            if add_idx in current_set:
                continue
            new_sum = current_sum + reciprocals[add_idx]
            if new_sum < 1:
                new_deficit = 1 - new_sum
                if new_deficit < best_deficit_final:
                    best_indices_final.append(add_idx)
                    best_deficit_final = new_deficit
                    improved = True
                    break
        if improved:
            continue
            
        # Try 1-swap
        for remove_idx in list(current_set):
            for add_idx in range(n):
                if add_idx in current_set:
                    continue
                new_sum = current_sum - reciprocals[remove_idx] + reciprocals[add_idx]
                if new_sum < 1:
                    new_deficit = 1 - new_sum
                    if new_deficit < best_deficit_final:
                        best_indices_final = [i for i in best_indices_final if i != remove_idx]
                        best_indices_final.append(add_idx)
                        best_deficit_final = new_deficit
                        improved = True
                        break
            if improved:
                break
    
    # Final check
    if not best_indices_final:
        return [0]  # Just 1/2, deficit = 1/2
    
    return best_indices_final
# EVOLVE-BLOCK-END
