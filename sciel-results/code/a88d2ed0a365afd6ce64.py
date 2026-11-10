# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    deadline = time.time() + 80  # Leave 10s margin
    selected = []
    # sums_by_size[k] is a set of sums of subsets of 'selected' of size k
    sums_by_size = {0: {0}}
    
    # Precompute or maintain sums_excl?
    # Let's just use a helper to check if target is a sum of j elements from a list.
    # For small lists, combinations is fine.
    # But we want to be fast.
    
    for x in range(1, 513):
        if time.time() > deadline:
            break
            
        # Check 1: x is not the mean of any subset of 'selected'
        valid = True
        max_k = len(selected)
        for k in range(2, max_k + 1):
            if k * x in sums_by_size.get(k, set()):
                valid = False
                break
        
        if not valid:
            continue
            
        # Check 2: No a in 'selected' is the mean of any subset containing x
        # This means for any a in selected, and any subset T of selected \ {a},
        # mean(T U {x}) != a
        # sum(T) + x = (len(T) + 1) * a
        # sum(T) = (len(T) + 1) * a - x
        for a in selected:
            others = [y for y in selected if y != a]
            # We need to check if there exists a subset T of others such that
            # sum(T) = (len(T) + 1) * a - x
            # Let j = len(T). Then sum(T) = (j + 1) * a - x.
            # j can range from 1 to len(others).
            for j in range(1, len(others) + 1):
                target_sum = (j + 1) * a - x
                # Check if target_sum can be formed by j elements from others
                # Use a simple DP or combinations if others is small
                # Since len(others) <= 47, combinations might be slow if we do it for every x.
                # Let's use a DP check for subset sum of size j.
                
                # Optimization: If target_sum is negative or too large, skip
                min_possible_sum = sum(sorted(others)[:j])
                max_possible_sum = sum(sorted(others)[-j:])
                if target_sum < min_possible_sum or target_sum > max_possible_sum:
                    continue
                    
                # Check if target_sum is in subset sums of 'others' of size j
                # We can precompute subset sums for 'others'?
                # 'others' changes as 'selected' changes.
                # Let's just use a recursive check or a set-based DP for 'others'.
                
                # Since this is called for every candidate x and every a, it might be slow.
                # Let's try to precompute subset sums for 'selected' and 'selected' without 'a'.
                
                # Alternative: Just use combinations if len(others) is small.
                # If len(others) > 20, it might be slow.
                
                # Let's use a DP for subset sum of size j from 'others'.
                # dp[i][k] = set of sums of size k from first i elements?
                # Or just a set of sums for each size.
                
                # Given the constraints, let's try a faster check.
                # We can maintain `sums_excl` for the current `selected`.
                
                # Let's refactor to maintain `sums_excl`.
                pass # Placeholder

        # This approach is getting complicated. Let's try a simpler, possibly slower but correct approach first,
        # and optimize if needed.
        
        # Simple Check 2:
        # For each a in selected, check if a is mean of any subset of selected + [x] containing x.
        # This is equivalent to: is there a subset T of selected \ {a} such that sum(T) + x = (len(T)+1)*a?
        
        # Let's just use a helper function that uses DP for subset sums of a given list.
        
        def is_subset_sum_of_size(lst, target, size):
            if size == 0:
                return target == 0
            if size > len(lst):
                return False
            if target < sum(sorted(lst)[:size]) or target > sum(sorted(lst)[-size:]):
                return False
            
            # DP
            # dp[k] is a set of sums of size k
            dp = [set() for _ in range(size + 1)]
            dp[0] = {0}
            for val in lst:
                for k in range(size, 0, -1):
                    if dp[k-1]:
                        new_sums = {s + val for s in dp[k-1]}
                        dp[k].update(new_sums)
            return target in dp[size]

        valid = True
        for a in selected:
            others = [y for y in selected if y != a]
            for j in range(1, len(others) + 1):
                target_sum = (j + 1) * a - x
                if is_subset_sum_of_size(others, target_sum, j):
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            selected.append(x)
            # Update sums_by_size
            new_sums = {k: set(s) for k, s in sums_by_size.items()}
            for k in range(1, len(selected) + 1):
                if k-1 in new_sums:
                    new_sums[k] = {s + x for s in new_sums[k-1]}
                    if k in new_sums:
                        new_sums[k].update(new_sums[k-1] if k-1 != k else set()) # Wait, standard 0/1 knapsack style
            # Correct update:
            new_sums = {k: set() for k in range(len(selected) + 1)}
            new_sums[0] = {0}
            # Rebuild from scratch? No, update incrementally.
            # old_sums = sums_by_size
            # new_sums[k] = old_sums[k] U {s + x for s in old_sums[k-1]}
            
            old_sums = sums_by_size
            new_sums = {k: set() for k in range(len(selected) + 1)}
            new_sums[0] = {0}
            for k in range(1, len(selected) + 1):
                if k in old_sums:
                    new_sums[k].update(old_sums[k])
                if k-1 in old_sums:
                    new_sums[k].update(s + x for s in old_sums[k-1])
            sums_by_size = new_sums

    return selected
# EVOLVE-BLOCK-END
