import time
from math import gcd
from functools import reduce

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    start_time = time.time()
    timeout = 85  # Safety margin for 90s limit
    
    # Generate all multiples up to a reasonable limit
    # The optimal solution is likely small. Let's go up to 10000 to be safe.
    limit = 10000
    all_mults = []
    for i, d in enumerate(divisors):
        for m in range(d, limit + 1, d):
            all_mults.append((m, i))
    
    # Sort by value
    all_mults.sort(key=lambda x: x[0])
    
    # We want to find the minimum range W
    # Iterate W from 9 upwards
    for W in range(9, 100):
        if time.time() - start_time > timeout:
            break
            
        # Use sliding window on all_mults
        # We need a window [L, R] such that R - L <= W
        # And all divisors are represented in the window.
        
        # To do this efficiently, we can iterate over the start index 'left'
        # and find the max 'right' such that all_mults[right][0] - all_mults[left][0] <= W
        
        # However, we also need to ensure that for each divisor, there is at least one multiple in the window.
        
        # Optimization: Instead of checking every 'left', we can just check windows formed by the values.
        # Since the number of multiples is ~10000/6 + ... ~ 2000, O(N^2) is 4M, which is fast in Python if optimized.
        # But we can do better.
        
        # Let's try a different approach:
        # For each possible minimum value 'min_val' in all_mults:
        #   Find all multiples in [min_val, min_val + W]
        #   Check if all divisors are covered.
        #   If yes, run backtracking.
        
        # To avoid redundant checks, we can just iterate through unique values in all_mults.
        
        seen_starts = set()
        
        for i in range(len(all_mults)):
            min_val = all_mults[i][0]
            if min_val in seen_starts:
                continue
            seen_starts.add(min_val)
            
            max_val = min_val + W
            
            # Collect available multiples for each divisor in [min_val, max_val]
            available = [[] for _ in range(n)]
            
            # We need to find all multiples in this range.
            # Since all_mults is sorted, we can find the range of indices.
            # But it's easier to just iterate all_mults? No, that's O(N).
            # Let's use binary search to find the slice.
            
            # Find left index (first >= min_val) and right index (last <= max_val)
            # Since min_val is all_mults[i][0], left is i.
            # Find right using binary search.
            # Python bisect module.
            import bisect
            # We need to search in the list of values.
            # Let's create a list of just values for bisect.
            # This is O(N) memory and time.
            # Or just use the property that all_mults is sorted.
            
            # Let's just iterate from i until value > max_val.
            # This is efficient because W is small.
            
            j = i
            while j < len(all_mults) and all_mults[j][0] <= max_val:
                val, idx = all_mults[j]
                available[idx].append(val)
                j += 1
            
            # Check if all divisors have at least one multiple
            if any(len(available[k]) == 0 for k in range(n)):
                continue
            
            # Check distinct assignment
            # Sort indices by constraint tightness
            indices = list(range(n))
            indices.sort(key=lambda k: len(available[k]))
            
            used = set()
            choices = [0] * n
            
            def backtrack(idx):
                if idx == n:
                    return True
                k = indices[idx]
                for val in available[k]:
                    if val not in used:
                        used.add(val)
                        choices[k] = val
                        if backtrack(idx + 1):
                            return True
                        used.remove(val)
                return False
            
            if backtrack(0):
                result = [choices[k] for k in range(n)]
                # Verify
                if len(set(result)) == n:
                    for k in range(n):
                        if result[k] % divisors[k] != 0 or not (0 < result[k] <= 1000000):
                            return divisors[:] # Fallback if verification fails (should not)
                    return result
    
    # Fallback
    return divisors[:]
