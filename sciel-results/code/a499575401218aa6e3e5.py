import time
import itertools
from fractions import Fraction

def _get_bad_masks(x):
    """
    Find all subsets of {1..24} \ {x} that sum to 1/x.
    Returns a list of integer bitmasks.
    """
    target = Fraction(1, x)
    # Elements available
    elems = [y for y in range(1, 25) if y != x]
    # Sort by value descending to help pruning (larger 1/y first)
    elems.sort(key=lambda y: Fraction(1, y), reverse=True)
    
    bad_masks = []
    
    def dfs(start_idx, current_sum, current_mask):
        if current_sum == target:
            if current_mask != 0:
                bad_masks.append(current_mask)
            return
        
        # Pruning: if current sum is already greater than target, return
        if current_sum > target:
            return
            
        # Pruning: if we can't reach target even by adding all remaining, return
        # We can precompute suffix sums of 1/y for elems
        # But for now, simple check: if start_idx >= len(elems), return
        if start_idx >= len(elems):
            return
            
        # Optional: Check if sum of all remaining < target - current_sum
        # This requires precomputation. Let's add it.
        
        for i in range(start_idx, len(elems)):
            y = elems[i]
            val = Fraction(1, y)
            new_sum = current_sum + val
            if new_sum > target:
                # Since elems are sorted by 1/y descending, subsequent 1/y are smaller.
                # So new_sum might decrease? No, we are adding positive values.
                # Wait, the loop iterates i. 
                # If we include y_i, we move to i+1.
                # If new_sum > target, we should NOT include y_i.
                # But we should still try including y_{i+1} etc.
                # So we just continue to the next i? 
                # No, the standard subset sum DFS structure is:
                # 1. Try include i
                # 2. Try exclude i (which is handled by the loop moving to i+1)
                # If we include i and exceed target, we just don't recurse.
                # We DO continue the loop to try i+1.
                continue
                
            # Include y
            dfs(i + 1, new_sum, current_mask | (1 << (y - 1)))
            
    # Precompute suffix sums for better pruning
    n = len(elems)
    suffix_sums = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + Fraction(1, elems[i])
        
    def dfs_pruned(start_idx, current_sum, current_mask):
        if current_sum == target:
            if current_mask != 0:
                bad_masks.append(current_mask)
            return
            
        if current_sum > target:
            return
            
        if start_idx >= n:
            return
            
        # If current_sum + sum(remaining) < target, return
        if current_sum + suffix_sums[start_idx] < target:
            return
            
        for i in range(start_idx, n):
            y = elems[i]
            val = Fraction(1, y)
            new_sum = current_sum + val
            if new_sum > target:
                # Since 1/y is decreasing, if 1/y_i makes it too big, 
                # 1/y_{i+1} is smaller, so it *might* fit.
                # So we cannot break. We must continue.
                continue
            dfs_pruned(i + 1, new_sum, current_mask | (1 << (y - 1)))
            
    dfs_pruned(0, Fraction(0), 0)
    return bad_masks

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    # Precompute bad masks for all x
    bad_masks = {}
    for x in range(1, 25):
        bad_masks[x] = _get_bad_masks(x)
        
    # Optimization: If time is running out, just return a known valid small set
    # e.g., {24} is always valid? 
    # 1/24. Can it be sum of others? Min sum of 2 others is 1/23+1/22 > 1/24.
    # So {24} is valid.
    # {23, 24}? 1/23 sum of others? Only 24. 1/24 != 1/23.
    # 1/24 sum of others? Only 23. 1/23 != 1/24.
    # So {23, 24} is valid.
    # We can build a greedy valid set quickly as a fallback.
    
    def is_valid_mask(mask):
        # Check for each x in mask
        for x in range(1, 25):
            if (mask >> (x - 1)) & 1:
                # Check if any bad mask for x is subset of mask
                for bm in bad_masks[x]:
                    if (bm & mask) == bm:
                        return False
        return True

    best_mask = 0
    best_size = 0
    
    # Try sizes from 24 down to 1
    # We can stop if we find a valid set of a certain size?
    # The task says "Maximize cardinality". So we should find the max.
    # But we must return *a* set. If we find size 24, that's max possible.
    # If we find size 23, we should check if 24 is possible? 
    # Actually, just iterate down. The first valid set found at size k is a candidate.
    # But we need the MAX size. 
    # So we should check 24, then 23, etc.
    # If 24 is invalid, check 23. If 23 is valid, is it possible that a size 24 set exists? No, we already checked 24.
    # So the first valid size we find is the maximum.
    
    for k in range(24, 0, -1):
        if time.time() > deadline - 5:
            break
            
        # Generate combinations of size k
        # For k=24, 1 comb. k=23, 24 combs. k=22, 276. k=21, 2024.
        # This is very fast.
        
        found = False
        for comb in itertools.combinations(range(1, 25), k):
            mask = 0
            for x in comb:
                mask |= (1 << (x - 1))
                
            if is_valid_mask(mask):
                # Found a valid set of size k. Since we iterate k downwards, this is the max.
                return list(comb)
                
        # If no valid set of size k found, continue to k-1
        
    # Fallback: Greedy construction if combinatorial search fails or times out
    current_mask = 0
    # Try to add elements in some order
    # Order: Larger elements first? Smaller?
    # Let's try descending order (24, 23, ...)
    for x in range(24, 0, -1):
        trial = current_mask | (1 << (x - 1))
        if is_valid_mask(trial):
            current_mask = trial
            
    result = [x for x in range(1, 25) if (current_mask >> (x - 1)) & 1]
    
    # Ensure non-empty
    if not result:
        result = [24]
        
    return result
