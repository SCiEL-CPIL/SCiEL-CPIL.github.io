import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # Try to find a set of size k, starting from max possible (9) down to 1
    for target_size in range(9, 0, -1):
        result = _find_set(target_size, deadline)
        if result is not None:
            return result
            
    # Fallback: return a valid small set
    return [31, 32, 34]

def _find_set(target_size, deadline):
    # We search for a superincreasing sequence of length target_size
    # a_1 < a_2 < ... < a_k
    # a_i > sum(a_1...a_{i-1})
    # a_i <= 256
    
    # We'll use DFS.
    # State: current list of numbers, current max sum, next index
    # To bias towards finding a solution, we can try larger numbers first?
    # Actually, smaller numbers leave more room for larger numbers?
    # No, superincreasing forces rapid growth.
    # To maximize the chance of fitting in 256, we should keep numbers as small as possible?
    # If we keep them small, the sum grows slowly, allowing more numbers.
    # But small numbers create more divisibility issues.
    # Let's try picking numbers in a range.
    
    def dfs(current_set, current_sum, idx):
        if time.time() > deadline:
            return None
        if idx == target_size:
            return list(current_set)
        
        # Minimum next value to satisfy superincreasing
        min_val = current_sum + 1
        
        # Maximum next value is 256
        # Also, we need enough room for the remaining (target_size - idx - 1) elements.
        # The remaining elements must be superincreasing starting from min_val.
        # The smallest possible sequence of length m starting from min_val is min_val, 2*min_val, 4*min_val...
        # Actually, if we pick x, the next min is x+current_sum.
        # To fit remaining r elements, we need to be able to pick r numbers.
        # The largest number in the final set will be at least ...
        
        # Let's just iterate candidates.
        # Range of candidates for current element:
        # Lower bound: min_val
        # Upper bound: 256
        
        # Heuristic: Try values in a specific order.
        # If we pick too small, we might get stuck with divisibility.
        # If we pick too large, we might exceed 256 for subsequent elements.
        
        # Let's try candidates from min_val up to 256.
        # To speed up, we can skip candidates that are obviously bad.
        
        # Optimization: The set of subset sums needs to be maintained.
        # current_sums is a set of all subset sums of current_set.
        
        # Pass current_sums to avoid recomputation.
        # But current_sums changes.
        
        # Let's structure the DFS to carry the set of sums.
        # For size 9, max 511 sums. Set operations are fast.
        
        # We need to pass the set of sums.
        # But we are inside a nested function, so we can use nonlocal or pass as arg.
        # Passing as arg is cleaner.
        
        # Actually, _find_set doesn't have current_sums.
        # Let's restart the DFS logic inside _find_set with proper state.
        
        pass # Placeholder

    # Re-implementing _find_set properly
    
    def _dfs(current_set, current_sum, current_sums, idx):
        if time.time() > deadline:
            return None
        if idx == target_size:
            return list(current_set)
        
        min_val = current_sum + 1
        if min_val > 256:
            return None
            
        # Iterate candidates
        # We can try to be smart about the range.
        # If we have r elements left, the last element must be at least min_val * 2^(r-1) ?
        # No, min_val is the next element.
        # Let's just loop.
        
        # To improve performance, we can sample or use a specific order.
        # Randomized search? No, deterministic is better for reproducibility, but we can try different orders.
        # Let's try from min_val upwards.
        
        for val in range(min_val, 257):
            if time.time() > deadline:
                return None
                
            # Check if val is valid with current_sums
            # New sums: val + s for s in current_sums U {0}
            new_sums = {val + s for s in current_sums}
            new_sums.add(val) # val + 0
            
            all_sums = current_sums.union(new_sums)
            
            # Check divisibility
            if _is_divisibility_free(all_sums):
                # Recursive step
                res = _dfs(current_set + [val], current_sum + val, all_sums, idx + 1)
                if res is not None:
                    return res
            
        return None

    # Initial state
    # Empty set, sum 0, sums {0} ? No, non-empty subset sums.
    # If current_set is empty, current_sums is empty set?
    # The problem says "Form every NONEMPTY subset sum".
    # So for empty set, there are no sums.
    # But when we add first element v, sums are {v}.
    # So initial current_sums = set()
    
    res = _dfs([], 0, set(), 0)
    return res

def _is_divisibility_free(sums):
    # sums is a set of integers
    # Check if any s1 divides s2 for s1 != s2
    # Sort sums
    s_list = sorted(sums)
    # For each s, check if 2s, 3s... is in sums
    # Since sums are positive, s < 2s.
    # We can iterate through s_list and check multiples.
    # Or just check pairs.
    # With ~500 sums, O(N^2) is 250k checks. Fast enough?
    # 250k integer divisions/modulos in Python might take ~0.05s.
    # If we do this at every node of the DFS, and DFS has many nodes, it could be slow.
    # But we prune early.
    # Let's try to optimize the check.
    
    # Better check:
    # For each s in s_list:
    #   for multiple in range(2*s, max_sum+1, s):
    #       if multiple in sums: return False
    # This is O(N * (Max/S_min)).
    # S_min is at least 1. Max is ~500*256 = 128000? No, max sum is sum of elements <= 9*256 = 2304.
    # Actually, max sum is sum of the set.
    # If elements are superincreasing, sum is approx 2 * max_element.
    # So max_sum is small.
    
    max_sum = max(sums) if sums else 0
    for s in s_list:
        if s == 0: continue
        # Check multiples
        m = 2 * s
        while m <= max_sum:
            if m in sums:
                return False
            m += s
    return True
