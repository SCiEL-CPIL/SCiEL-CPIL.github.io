"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
import random

def construct():
    # Compute budget guard
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin for 90s limit
    
    denominators = list(range(2, 49))
    n = len(denominators)
    recip = [Fraction(1, d) for d in denominators]
    
    # We want to find a subset S of indices such that sum_{i in S} recip[i] < 1
    # and 1 - sum is minimized.
    
    best_indices = []
    best_deficit = Fraction(1, 1)  # Max possible deficit is 1 (empty set)
    
    # Helper to update best
    def update_best(indices, total):
        nonlocal best_indices, best_deficit
        if not indices:
            return
        deficit = 1 - total
        if deficit > 0 and deficit < best_deficit:
            best_deficit = deficit
            best_indices = list(indices)
    
    # Generate diverse initial solutions using greedy with randomization
    # Strategy: Sort by denominator descending (largest fractions first) or random
    # Greedy: Add if sum + recip < 1
    
    initializations = []
    
    # 1. Standard greedy (smallest denominators first -> largest fractions)
    # This is what the seed does, but let's make sure we capture it.
    # Actually, to get close to 1, we often want to pack tightly.
    # Greedy by size (1/2, 1/3, ...) is a good start.
    
    for attempt in range(20):
        if time.time() > deadline:
            break
            
        # Randomize the order of consideration to get different subsets
        # But keep some structure. 
        # Idea: Sort by 1/d with some noise, or just random permutation.
        # Random permutation might yield poor sums. 
        # Better: Greedy but with random tie-breaking or random skipping?
        
        # Let's use a "Greedy with Random Restart" approach.
        # Sort indices by reciprocal value descending (1/2 > 1/3 > ...)
        # But sometimes skipping a large one allows fitting more smaller ones closer to 1.
        
        # Strategy A: Greedy by size (descending 1/d)
        # Strategy B: Random subset generation near the "frontier"
        
        # Let's stick to local search from greedy starts.
        
        # Generate a greedy solution
        # To diversify, we can randomly exclude some candidates in the greedy phase
        # or use a different sorting criterion.
        
        # Let's try sorting by 1/d descending, but with random noise in the "value"
        # or just random permutation of indices, then greedy add.
        
        indices = list(range(n))
        
        if attempt == 0:
            # Deterministic greedy: largest fractions first
            pass # already sorted by index which is ascending d, so 1/d descending
        else:
            # Randomize the order of consideration
            random.shuffle(indices)
            # But pure random order greedy is bad. 
            # Let's mix: sort by 1/d but with random noise?
            # Or just shuffle and then sort by 1/d but with random swaps?
            
            # Simpler: Just shuffle. The greedy "add if fits" on a shuffled list
            # produces a random valid subset.
            pass
            
        current_subset = set()
        current_sum = Fraction(0)
        
        for i in indices:
            if current_sum + recip[i] < 1:
                current_subset.add(i)
                current_sum += recip[i]
        
        update_best(list(current_subset), current_sum)
        
        # Local Search: 1-swap and 2-swap
        # Time check
        if time.time() > deadline:
            break
            
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1-swap: Remove i, add j (i in subset, j not in subset)
            # Gain = recip[j] - recip[i]
            # We want to maximize gain while keeping sum < 1.
            # New sum = current_sum - recip[i] + recip[j] < 1
            # => recip[j] < 1 - current_sum + recip[i]
            
            # Precompute for efficiency? n=47, so O(n^2) is fine.
            # List of items in subset
            in_sub = list(current_subset)
            not_in_sub = [i for i in range(n) if i not in current_subset]
            
            best_swap = None
            best_gain = Fraction(0)
            
            for i in in_sub:
                for j in not_in_sub:
                    new_sum = current_sum - recip[i] + recip[j]
                    if new_sum < 1:
                        gain = recip[j] - recip[i]
                        if gain > best_gain:
                            best_gain = gain
                            best_swap = (i, j)
            
            if best_swap:
                i, j = best_swap
                current_subset.remove(i)
                current_subset.add(j)
                current_sum += best_gain
                improved = True
                update_best(list(current_subset), current_sum)
                
            # 2-swap: Remove i1, i2, add j1, j2
            # This is O(n^4) which is 47^4 ~ 4.8M. Might be too slow per iteration.
            # Let's stick to 1-swap for now, it's proven to work well in the prompt's hint.
            # The hint said "Extensive 1-swap local search ... should find much smaller deficits".
            
            # If no 1-swap improves, try random 2-swaps?
            # Or just stop if no 1-swap improves (local minimum).
            
            if not improved:
                break
                
    if not best_indices:
        # Fallback: empty set is invalid (non-empty required). 
        # But greedy should always find at least {0} (1/2 < 1).
        # If something went wrong, return [0]
        best_indices = [0]
        
    return best_indices
# EVOLVE-BLOCK-END
