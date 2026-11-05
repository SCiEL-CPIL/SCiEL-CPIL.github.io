"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
import numpy as np

N = 256

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s kill
    
    # Precompute conflict matrix:
    # conflict_mask[i] is a bitmask of indices j such that adding j to a set containing i
    # would violate the Sidon property (specifically, the sums involving i and j would clash
    # with sums involving i and i, or j and j, or other existing elements).
    # Actually, for a set S, adding x is valid iff for all a in S:
    #   1. 2x not in sums(S)
    #   2. x+a not in sums(S) for all a in S
    #   3. x+a != x+b for a!=b (always true since a!=b)
    #   4. x+a != a+a (i.e., x != a, true since x not in S)
    #   5. x+a != b+b for b in S (i.e., x+a != 2b)
    
    # Let's precompute a matrix M where M[i][j] is True if i and j cannot coexist in a Sidon set.
    # i and j cannot coexist if:
    #   2i == 2j (i==j)
    #   2i == i+j => i==j
    #   2j == i+j => i==j
    # Wait, that's not right.
    # A set {i, j} is Sidon iff all sums i+i, i+j, j+j are distinct.
    # i+i != i+j => i != j
    # i+i != j+j => i != j
    # i+j != j+j => i != j
    # So any pair of distinct integers is a valid Sidon set of size 2.
    # The conflict arises when adding a third element k.
    # {i, j, k} is Sidon iff all sums i+i, i+j, i+k, j+j, j+k, k+k are distinct.
    
    # For a given set S, x is addable iff:
    #   2x is not in {2a | a in S}
    #   x+a is not in {a+b | a,b in S} for all a in S
    # This is equivalent to checking if the new sums {2x, x+a for a in S} are disjoint from existing sums {a+b for a,b in S}.
    
    # Let's represent the set of sums for a given subset as a bitmask.
    # The maximum sum is 256+256=512. We need a bitmask of size 513 (indices 0..512).
    # Python integers can handle this easily.
    
    # Precompute sum_mask[i] for a set containing only i:
    # sum_mask[i] = 1 << (2*i)
    # When adding j to a set S with sum_mask S_mask:
    # new_sums = {2j} U {j+i for i in S}
    # We need new_sums to be disjoint from S_mask.
    
    # Let's precompute a matrix of "conflict with existing sums".
    # This is hard to precompute statically for all subsets.
    
    # Alternative: Use a fast greedy with bitmasks.
    # State: current_set (list), current_sum_mask (int)
    # To check if x is addable:
    #   new_sums_mask = (1 << (2*x))
    #   for a in current_set:
    #       new_sums_mask |= (1 << (x+a))
    #   if new_sums_mask & current_sum_mask:
    #       conflict
    #   else:
    #       addable
    
    # This check is O(|S|) per candidate.
    # To find a small maximal set, we want to pick elements that block many others.
    
    # Let's try a different approach: 
    # 1. Generate a random maximal Sidon set using a fast greedy.
    # 2. Try to remove elements while maintaining maximality.
    #    Removing x is allowed if the resulting set is still maximal.
    #    Maximality check: for all y not in S, y is blocked.
    #    y is blocked if adding y causes a conflict.
    
    # This "Remove" strategy is often more effective for finding small maximal sets.
    
    def get_sum_mask(selected):
        mask = 0
        for i in selected:
            mask |= (1 << (2*i))
        for i in range(len(selected)):
            for j in range(i+1, len(selected)):
                mask |= (1 << (selected[i] + selected[j]))
        return mask

    def is_addable(x, selected, sum_mask):
        new_mask = 1 << (2*x)
        for a in selected:
            new_mask |= (1 << (x+a))
        return (new_mask & sum_mask) == 0

    def is_maximal(selected, sum_mask):
        sel_set = set(selected)
        for x in range(1, N+1):
            if x in sel_set:
                continue
            if is_addable(x, selected, sum_mask):
                return False
        return True

    def fast_greedy_maximal():
        selected = []
        sum_mask = 0
        remaining = list(range(1, N+1))
        random.shuffle(remaining)
        
        for x in remaining:
            if is_addable(x, selected, sum_mask):
                selected.append(x)
                sum_mask |= (1 << (2*x))
                for a in selected[:-1]:
                    sum_mask |= (1 << (x+a))
        
        return selected, sum_mask

    def try_remove(selected, sum_mask):
        # Try to remove elements to minimize size while maintaining maximality
        # We can only remove x if the new set is still maximal.
        # This is expensive to check for every removal.
        # Instead, we can try to remove elements in a random order and check maximality.
        
        current_selected = selected[:]
        current_sum_mask = sum_mask
        current_set = set(current_selected)
        
        # Sort by some heuristic? Maybe smaller elements are more "redundant"?
        # Or just random order.
        
        order = current_selected[:]
        random.shuffle(order)
        
        for x in order:
            if x not in current_set:
                continue
            
            # Temporarily remove x
            new_selected = [y for y in current_selected if y != x]
            
            # Recompute sum_mask? 
            # Removing x removes sums: 2x, and x+a for all a in new_selected.
            # But we also need to ensure that the remaining sums are still distinct (they are, since subset).
            
            # Check if new set is maximal.
            # This requires checking all y not in new_selected.
            # If any y is addable, we cannot remove x.
            
            # Optimization: Only check y that were previously blocked by x.
            # But identifying those is also expensive.
            
            # Let's just check maximality directly.
            # To speed up, we can precompute the sum_mask of new_selected.
            
            # Actually, recomputing sum_mask from scratch is O(k^2) where k is size.
            # k is around 15-20. So 400 ops.
            # Maximality check is O(N * k) = 256 * 20 = 5120 ops.
            # Total per removal attempt: ~5000 ops.
            # 20 removal attempts: 100,000 ops.
            # This is very fast in Python if optimized, but let's use bitmasks.
            
            # Let's just do the check.
            new_sum_mask = get_sum_mask(new_selected)
            
            if is_maximal(new_selected, new_sum_mask):
                current_selected = new_selected
                current_sum_mask = new_sum_mask
                current_set.discard(x)
        
        return current_selected

    best_result = None
    best_size = float('inf')
    
    # Run multiple iterations
    for _ in range(500):
        if time.time() > deadline:
            break
            
        selected, sum_mask = fast_greedy_maximal()
        
        # Try to reduce size
        reduced = try_remove(selected, sum_mask)
        
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_result = reduced[:]
            
        # Early exit if we find a very small set
        if best_size <= 10:
            break

    return best_result
# EVOLVE-BLOCK-END
