from fractions import Fraction
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 60.0  # seconds
    
    # We want to maximize cardinality.
    # Greedy strategy: add largest possible elements first?
    # Or smallest?
    # If we add smallest first, we might block larger ones?
    # No, the condition is symmetric in a way, but "other elements" matters.
    # If we have {1, 2}, 1/1 = 1, 1/2 = 0.5.
    # 1 != 0.5. 0.5 != 1. Valid.
    # If we have {1, 2, 3}.
    # 1/1 = 1. Subsets of {2,3}: 1/2, 1/3, 1/2+1/3=5/6. None is 1.
    # 1/2 = 0.5. Subsets of {1,3}: 1, 1/3, 1+1/3. None is 0.5.
    # 1/3. Subsets of {1,2}: 1, 1/2, 1.5. None is 1/3.
    # So {1,2,3} is valid.
    
    # Let's try to build the set by adding elements from 24 down to 1.
    # Why? Because larger elements have smaller reciprocals,
    # making it harder to form sums (they are "small" contributions).
    # Actually, if we pick large numbers, the sums of their reciprocals are small.
    # The targets are also small.
    # The "collision" risk is higher when reciprocals are comparable.
    # If we pick 1, 1/1=1 is a large target.
    # If we pick 24, 1/24 is a small target.
    
    # Let's stick to the previous insight:
    # If we pick S = {12, ..., 24}, it is valid.
    # Can we add 11?
    # We check if 1/11 is in the subset sums of {12, ..., 24}.
    
    S = set()
    
    # We iterate from 24 down to 1
    # If we add 24, 23, ..., 12, we are safe by the min-sum argument.
    # We can just add them directly.
    for x in range(24, 11, -1):
        S.add(x)
        
    # Now try to add 11, 10, ..., 1
    # For each candidate x, check if 1/x can be formed by a subset of S.
    # If yes, skip x. If no, add x.
    
    for x in range(11, 0, -1):
        if time.time() > deadline - 5:
            break
            
        target = Fraction(1, x)
        elements = [Fraction(1, y) for y in S]
        
        # Check if target can be formed by a non-empty subset of elements
        # Use DFS with pruning
        # Sort elements in descending order to prune faster?
        # If we take large reciprocals first, we exceed target faster.
        elements.sort(reverse=True)
        
        can_form = False
        n = len(elements)
        
        # Memoization or just DFS?
        # DFS with pruning:
        # State: index, current_sum
        # If current_sum > target: return False
        # If index == n: return current_sum == target
        # Try skip, try take.
        
        # To avoid deep recursion limits or overhead, we can use a set of reachable sums?
        # No, sums are Fractions.
        
        # Let's use a simple recursive DFS.
        # Since n is at most 23, and pruning is effective, it should be fine.
        
        def dfs(idx, curr_sum):
            if curr_sum > target:
                return False
            if idx == n:
                return curr_sum == target and curr_sum != 0 # Non-empty subset
            # Option 1: skip
            if dfs(idx + 1, curr_sum):
                return True
            # Option 2: take
            if dfs(idx + 1, curr_sum + elements[idx]):
                return True
            return False
        
        # Optimization: if sum of all remaining < target - curr_sum, return False
        # Precompute suffix sums?
        suffix_sums = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix_sums[i] = suffix_sums[i + 1] + elements[i]
            
        def dfs_opt(idx, curr_sum):
            if curr_sum > target:
                return False
            if idx == n:
                return curr_sum == target and curr_sum != 0
            # Prune: if curr_sum + suffix_sums[idx] < target, impossible to reach target
            if curr_sum + suffix_sums[idx] < target:
                return False
            # Option 1: take
            if dfs_opt(idx + 1, curr_sum + elements[idx]):
                return True
            # Option 2: skip
            if dfs_opt(idx + 1, curr_sum):
                return True
            return False
        
        can_form = dfs_opt(0, Fraction(0, 1))
        
        if not can_form:
            S.add(x)
            
    return list(S)

# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    import time
    
    start_time = time.time()
    deadline = 60.0
    
    S = set()
    
    # Add 12..24 directly as they are safe
    for x in range(24, 11, -1):
        S.add(x)
        
    for x in range(11, 0, -1):
        if time.time() > deadline - 5:
            break
            
        target = Fraction(1, x)
        elements = [Fraction(1, y) for y in S]
        elements.sort(reverse=True)
        n = len(elements)
        
        suffix_sums = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix_sums[i] = suffix_sums[i + 1] + elements[i]
            
        def dfs_opt(idx, curr_sum):
            if curr_sum > target:
                return False
            if idx == n:
                return curr_sum == target and curr_sum != 0
            if curr_sum + suffix_sums[idx] < target:
                return False
            if dfs_opt(idx + 1, curr_sum + elements[idx]):
                return True
            if dfs_opt(idx + 1, curr_sum):
                return True
            return False
        
        if not dfs_opt(0, Fraction(0, 1)):
            S.add(x)
            
    return list(S)
# EVOLVE-BLOCK-END
