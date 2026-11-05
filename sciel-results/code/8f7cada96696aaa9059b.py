from fractions import Fraction
import time

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin
    
    # We want to select a subset of [1, 24]
    # Start with an empty set and try to add elements from 24 down to 1
    # This prioritizes larger numbers (smaller reciprocals) which are less likely to form sums
    # Actually, adding larger numbers first creates a set of small reciprocals.
    # When we try to add a smaller number (larger reciprocal), we check if that large reciprocal
    # can be formed by the sum of the existing small reciprocals.
    # Since the existing reciprocals are small, their sums are small.
    # The target is large.
    # So it's unlikely to be formed.
    
    selected = []
    
    # Helper to check if target is a subset sum of items
    def is_subset_sum(target, items):
        # items is a list of Fractions
        # Use DFS with pruning
        # Sort items descending to find large sums quickly? 
        # Actually, for subset sum, sorting descending helps if we subtract.
        # Let's use a simple recursive DFS with memoization on (idx, current_sum)?
        # Current sum can be many values.
        # Given small N, simple DFS is fine.
        
        n = len(items)
        if n == 0:
            return target == 0
        
        # Sort items descending
        sorted_items = sorted(items, reverse=True)
        
        # Precompute suffix sums for pruning
        suffix_sums = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix_sums[i] = suffix_sums[i + 1] + sorted_items[i]
            
        def dfs(idx, current_sum):
            if current_sum == target:
                return True
            if idx == n:
                return False
            # Pruning 1: if current sum exceeds target (items are positive)
            if current_sum > target:
                return False
            # Pruning 2: if current sum + max possible remaining sum < target
            if current_sum + suffix_sums[idx] < target:
                return False
            
            # Option 1: Include item idx
            if dfs(idx + 1, current_sum + sorted_items[idx]):
                return True
            # Option 2: Exclude item idx
            if dfs(idx + 1, current_sum):
                return True
            return False
        
        return dfs(0, Fraction(0))

    # Iterate from 24 down to 1
    for x in range(24, 0, -1):
        if time.time() > deadline:
            break
            
        # Check if 1/x is a subset sum of reciprocals of 'selected'
        # If yes, we cannot add x (because 1/x would be a sum of others)
        # If no, we can add x
        
        reciprocals = [Fraction(1, y) for y in selected]
        target = Fraction(1, x)
        
        # Optimization: If selected is empty, target != 0, so add.
        # If selected is not empty, check.
        
        # Quick check: If target > sum of all reciprocals, it's impossible.
        # If target < min reciprocal? No, target is 1/x. Min reciprocal is 1/max(selected).
        # Since x < min(selected) (we are adding smaller x to larger selected),
        # 1/x > 1/max(selected). So target is larger than any single item.
        # It could be a sum of multiple items.
        
        can_add = not is_subset_sum(target, reciprocals)
        
        if can_add:
            selected.append(x)
            
    # Ensure non-empty
    if not selected:
        return [24]
        
    return selected
