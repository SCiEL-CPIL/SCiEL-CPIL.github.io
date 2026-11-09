import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit
    
    n = 12
    m = 40
    total_masks = 1 << n
    
    # Precompute popcount? Not strictly necessary for comparability check, but good for debugging.
    # We need to select 'm' distinct masks.
    
    # Initial family: Empty set and Full set are always good anchors.
    # Empty set (0) is subset of all.
    # Full set ((1<<n)-1) is superset of all.
    # They are comparable to each other.
    
    # Strategy:
    # 1. Start with a few strong candidates.
    # 2. Greedily add the next best candidate.
    
    # To improve results, we can try a few different initial seeds or random restarts if time permits.
    # But greedy from {0, full} is a strong baseline.
    
    # Let's try a slightly more robust greedy:
    # Instead of just adding the best, maybe we should consider that adding a set of size k 
    # makes it comparable to existing sets of size < k and > k.
    
    # We'll run a single greedy pass. It's fast.
    
    # List of all masks
    # We don't need to store all masks in a list if we can iterate 0..4095.
    
    selected = []
    # Start with 0 and full
    selected.append(0)
    selected.append(total_masks - 1)
    
    # We need m - 2 more sets
    # To make it deterministic and robust, we iterate candidates.
    
    # Optimization: 
    # For each candidate c, score = sum(1 for s in selected if comparable(c, s))
    # comparable(c, s) is (c & s) == c or (c & s) == s
    
    # We can speed this up by noting that if we have many sets of a certain size,
    # we should pick sets that are subsets of large sets and supersets of small sets.
    
    # Let's just do the greedy.
    # 38 iterations.
    # Each iteration checks ~4096 candidates.
    # Each check involves ~38 comparisons (bit ops).
    # 38 * 4096 * 38 ~ 5.8 million ops. Very fast.
    
    for _ in range(m - len(selected)):
        if time.time() > deadline:
            break
            
        best_score = -1
        best_mask = -1
        
        # Iterate all possible masks
        # We can skip masks already in selected. 
        # Since selected is small, checking `if mask in selected` is O(k).
        # Alternatively, maintain a set of selected masks for O(1) lookup.
        
        selected_set = set(selected)
        
        for mask in range(total_masks):
            if mask in selected_set:
                continue
                
            score = 0
            # Check comparability with all selected
            for s in selected:
                # Strict subset check
                # c subset s: (c & s) == c and c != s
                # s subset c: (s & c) == s and s != c
                # Since mask != s is guaranteed by the set check, we just check intersection.
                if (mask & s) == mask or (mask & s) == s:
                    score += 1
                    
            if score > best_score:
                best_score = score
                best_mask = mask
                
        if best_mask != -1:
            selected.append(best_mask)
        else:
            break

    # Convert to lists of integers
    # Requirement: "each as a list of distinct integers"
    # "integers must be Python int"
    
    result = []
    for mask in selected:
        subset = []
        for i in range(n):
            if mask & (1 << i):
                subset.append(i)
        result.append(subset)
        
    # Ensure exactly 40 lists. 
    # If the loop broke early due to time, we might have fewer.
    # The problem says "exactly forty". 
    # If we run out of time, we should fill with valid distinct sets to maintain validity, 
    # even if score is lower, rather than returning invalid count.
    # But 5.8M ops is < 0.1s in Python. Time limit is not a risk for this specific algorithm.
    
    # Just in case, if len(result) < 40, add random unused masks.
    if len(result) < 40:
        used_masks = set(selected)
        # Shuffle all masks and pick unused ones
        all_masks = list(range(total_masks))
        random.shuffle(all_masks)
        for mask in all_masks:
            if len(result) >= 40:
                break
            if mask not in used_masks:
                used_masks.add(mask)
                subset = [i for i in range(n) if mask & (1 << i)]
                result.append(subset)
                
    return result
