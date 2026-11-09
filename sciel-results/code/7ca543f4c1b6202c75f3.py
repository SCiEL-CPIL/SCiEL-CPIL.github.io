"""Rewritten program: Branch and Bound with tight upper bounds for reciprocal-sum maximization."""
# EVOLVE-BLOCK-START
import time
from math import lcm, gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before 90s kill

    # Precompute LCM table for all pairs in [2, 256]
    # lcm(a,b) > 256 means the pair is "compatible"
    n = 255  # numbers 2..256
    # Index: i corresponds to number i+2
    # compatible[i][j] = True if lcm(i+2, j+2) > 256
    
    # We'll use a bitmask approach or list-based B&B
    # Numbers sorted by 1/x descending (i.e., x ascending) for tighter bounds
    nums = list(range(2, 257))
    # Sort by reciprocal descending = by value ascending
    # Actually for B&B we want to consider high-value items first
    # 1/x is larger for smaller x, so we process small x first
    
    # Precompute compatibility: for each number, which other numbers are compatible
    # compatible_mask[i] = bitmask of j where lcm(nums[i], nums[j]) > 256
    # But 255 bits is large; let's use a set or boolean matrix
    
    # Boolean matrix: comp[i][j] = True if nums[i] and nums[j] are compatible (lcm > 255, i.e., > 256? No, > 256)
    # Wait: lcm(a,b) > 256. So lcm >= 257.
    
    comp = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            a, b = nums[i], nums[j]
            if lcm(a, b) > 256:
                comp[i][j] = True
                comp[j][i] = True
    
    # Precompute for each index, the set of compatible indices
    compat_sets = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and comp[i][j]:
                compat_sets[i].add(j)
    
    # Sort indices by 1/x descending (i.e., x ascending) for B&B ordering
    # Index i corresponds to nums[i] = i+2
    # 1/(i+2) is decreasing in i, so index 0 (x=2) has highest reciprocal
    # So we process indices in order 0, 1, 2, ..., n-1
    
    # Upper bound: sum of 1/x for all remaining candidates that are compatible with current selection
    # For a tighter bound, at each node we can compute the maximum additional sum
    
    # B&B state: (current_selection, next_index_to_consider)
    # We maintain:
    #   best_sum = Fraction
    #   best_set = list of selected numbers
    #   current_sum = Fraction
    #   current_set = list of selected indices
    #   compatible_remaining = set of indices that are compatible with all in current_set
    
    best_sum = Fraction(0, 1)
    best_set = []
    
    # For efficiency, track which indices are still "available" (compatible with current selection)
    # Start: all indices are available
    
    def upper_bound(available_indices, current_sum):
        """Compute an upper bound on the total sum if we could add any subset of available_indices.
        The tightest simple bound is current_sum + sum(1/x for x in available_indices).
        But we can do better: since all pairs in the final set must be mutually compatible,
        the additional sum is at most the sum over available indices, but we can tighten
        by noting that not all available indices are mutually compatible.
        
        A simple but effective bound: current_sum + sum(1/nums[i] for i in sorted(available_indices, key=lambda i: nums[i]))
        This is valid because any feasible extension is a subset of available_indices.
        """
        if not available_indices:
            return current_sum
        # Sum of reciprocals of all available indices
        add = sum(Fraction(1, nums[i]) for i in available_indices)
        return current_sum + add
    
    # Recursive B&B
    # To avoid deep recursion, we'll use an iterative approach or careful recursion
    # Depth is at most 255, which is fine for recursion if we set recursion limit
    
    import sys
    sys.setrecursionlimit(10000)
    
    def bb(current_idx, current_sum, current_set, available):
        """
        current_idx: next index to consider (we process in order 0,1,2,...)
        current_sum: Fraction, sum of 1/x for selected
        current_set: list of selected indices
        available: set of indices >= current_idx that are compatible with all in current_set
        """
        nonlocal best_sum, best_set
        
        if time.time() > deadline:
            return
        
        # Update best if current is better
        if current_sum > best_sum:
            best_sum = current_sum
            best_set = list(current_set)
        
        # Try to add more elements
        # Upper bound check
        if not available:
            return
        
        ub = upper_bound(available, current_sum)
        if ub <= best_sum:
            return
        
        # Process indices from current_idx onward
        for i in range(current_idx, n):
            if time.time() > deadline:
                return
            
            if i not in available:
                continue
            
            # Option 1: include i
            new_available = set(available)
            new_available.discard(i)
            # Filter: only keep indices compatible with i
            new_available = {j for j in new_available if comp[i][j]}
            
            new_sum = current_sum + Fraction(1, nums[i])
            new_set = current_set + [i]
            
            bb(i + 1, new_sum, new_set, new_available)
            
            # Option 2: exclude i (implicitly handled by the loop continuing)
            # But we need to prune: if even including i and all subsequent doesn't help, skip
            # The loop naturally handles exclusion by just not including i
            
            # Pruning: if the best possible from remaining (excluding i) can't beat best, break
            # remaining_after_i = available - {i} - {indices < i already processed}
            # Actually, available already only contains indices >= current_idx that are compatible
            # After excluding i, the remaining available for subsequent iterations is available minus {i}
            # But we also need to consider that subsequent indices must be compatible with current_set (which is unchanged)
            # So remaining = available - {i}
            remaining = available - {i}
            if remaining:
                ub_rem = upper_bound(remaining, current_sum)
                if ub_rem <= best_sum:
                    # Can't improve by excluding i and taking any subset of remaining
                    # But wait, we might have already found a better solution by including i
                    # The break condition: if even the best from remaining can't beat best_sum, 
                    # and we've already explored including i, then we can break
                    # But this isn't quite right because including i might have led to a good solution
                    # Actually, the standard B&B: we continue the loop, and the ub check at the top of the loop
                    # will prune. Let's just continue.
                    pass
    
    # Initial call: all indices available, start from index 0
    all_indices = set(range(n))
    bb(0, Fraction(0, 1), [], all_indices)
    
    # Convert best_set (indices) to numbers
    result = [nums[i] for i in best_set]
    
    if not result:
        # Fallback: at least one element
        result = [2]
    
    return result
# EVOLVE-BLOCK-END
