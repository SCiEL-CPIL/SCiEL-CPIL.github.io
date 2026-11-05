import time
import sys
from fractions import Fraction

def construct():
    start = time.time()
    deadline = start + 85  # 85 second budget, 5s margin
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM of 2..48
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
        
    def lcm(a, b):
        return a // gcd(a, b) * b
        
    L = 1
    for d in denominators:
        L = lcm(L, d)
        
    # Weights are integers
    weights = [L // d for d in denominators]
    
    # Split into two sets A and B.
    mid = n // 2
    A = weights[:mid]
    B = weights[mid:]
    
    # Generate all subset sums for A (values only)
    sums_A = [0]
    for w in A:
        # Use list comprehension for speed
        sums_A += [s + w for s in sums_A]
        
    # Generate all subset sums for B (values only)
    sums_B = [0]
    for w in B:
        sums_B += [s + w for s in sums_B]
        
    # Sort sums_A
    sums_A.sort()
    
    # We want to find max a + b < L
    # Two pointer approach
    # i points to largest in A, j iterates B
    # Since sums_A is sorted ascending, we start i at end.
    # As j increases (b increases), the max allowed a decreases.
    
    best_sum_val = -1
    best_a_val = 0
    best_b_val = 0
    
    i = len(sums_A) - 1
    for b_val in sums_B:
        # If b_val >= L, no a can make it < L (since a >= 0)
        if b_val >= L:
            break
            
        # Find largest a_val such that a_val < L - b_val
        # i is the index of the largest valid a_val for the previous (smaller) b_val.
        # Since b_val is increasing, L - b_val is decreasing.
        # So i can only decrease.
        while i >= 0 and sums_A[i] + b_val >= L:
            i -= 1
            
        if i < 0:
            break
            
        a_val = sums_A[i]
        total_val = a_val + b_val
        
        if total_val > best_sum_val:
            best_sum_val = total_val
            best_a_val = a_val
            best_b_val = b_val
            
        # Time check
        if time.time() > deadline - 5:
            break
            
    if best_sum_val < 0:
        return [0]
        
    # Now find the masks corresponding to best_a_val and best_b_val
    # We need to re-traverse the subsets to find one that matches the value.
    
    def find_mask(weights_subset, target_val):
        # Returns mask (integer) of the subset that sums to target_val
        # We can do this with a simple DFS or iterative generation
        # Since we just need ONE mask, we can stop early.
        
        # Iterative generation with early exit is tricky with the doubling method.
        # Let's use a simple recursive search or DFS.
        # n_subset is at most 24.
        
        # Actually, we can just regenerate the sums and keep the mask, 
        # but that defeats the memory saving.
        # However, we only do this ONCE.
        # So we can afford to store (sum, mask) for the search phase?
        # No, we already discarded them.
        
        # Let's do a DFS to find a subset summing to target_val.
        # This is fast because we prune heavily?
        # Or just iterate through all 2^k subsets? 2^24 is 16M.
        # 16M iterations in Python is ~1-2 seconds. Acceptable.
        
        k = len(weights_subset)
        # We can use bit manipulation to generate masks?
        # Or simple recursion.
        
        # Let's use a set of remaining sums?
        # No, just iterate.
        
        # Optimization: Sort weights descending for pruning in DFS?
        # But we need to return the mask relative to original indices.
        
        # Simple approach:
        # Generate all (sum, mask) for this subset, but stop when found?
        # We can't easily stop the doubling method early.
        
        # Let's just do a linear scan of all 2^k masks?
        # For k=24, 16M masks.
        # For each mask, calculate sum.
        # This is O(2^k * k) if done naively.
        # 16M * 24 = 400M ops. Too slow in Python.
        
        # Better: Use the same doubling method, but store masks, and check?
        # We can generate (sum, mask) pairs, but we don't store them all.
        # We just check if sum == target.
        
        pairs = [(0, 0)]
        for i, w in enumerate(weights_subset):
            new_pairs = []
            for s, m in pairs:
                new_s = s + w
                new_m = m | (1 << i)
                if new_s == target_val:
                    return new_m
                # Pruning: if new_s > target_val, we can skip?
                # Only if all weights are positive (which they are).
                # But we might need smaller sums later?
                # No, if we are looking for EXACT target, and weights are positive,
                # any sum > target cannot lead to target.
                # So we can prune sums > target.
                if new_s <= target_val:
                    new_pairs.append((new_s, new_m))
            pairs = new_pairs # We only keep valid partial sums?
            # Wait, if we prune, we might miss a combination where a large weight is added later?
            # No, if current sum > target, adding more positive weights keeps it > target.
            # So pruning is safe.
            
        # If not found in pruned search, maybe target is 0?
        if target_val == 0:
            return 0
            
        # Fallback: full search (should not happen if target was from our list)
        # This is a safety net
        for mask in range(1 << k):
            s = 0
            for i in range(k):
                if mask & (1 << i):
                    s += weights_subset[i]
            if s == target_val:
                return mask
                
        return 0 # Should not happen

    best_a_mask = find_mask(A, best_a_val)
    best_b_mask = find_mask(B, best_b_val)
    
    # Reconstruct indices
    selected = []
    for i in range(mid):
        if best_a_mask & (1 << i):
            selected.append(i)
    for i in range(n - mid):
        if best_b_mask & (1 << i):
            selected.append(mid + i)
            
    # Verify with exact fractions
    total_frac = sum(Fraction(1, denominators[k]) for k in selected)
    if total_frac < 1:
        return selected
    else:
        # Fallback
        return [0]
