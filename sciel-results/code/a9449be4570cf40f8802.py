import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin against 90s limit

    n = 30
    # Calculate LCM of 1..30
    L = 1
    for i in range(1, n + 1):
        L = L * i // gcd(L, i)
    
    weights = [L // i for i in range(1, n + 1)]
    
    # Split into two halves
    half = n // 2
    w1 = weights[:half]
    w2 = weights[half:]
    
    # Generate all sums for first half
    # sum -> coeffs (tuple of length half)
    sums1_map = {0: (0,) * half}
    
    # Iterative generation to avoid recursion depth issues and overhead
    # Current sums: dict {sum_val: coeffs_tuple}
    # Next sums: for each sum, add -w, 0, +w
    
    # To speed up, we can use a list of (sum, coeffs) and then deduplicate?
    # No, dict is better for O(1) lookup later.
    # But generating 14M entries in a dict in Python might be slow.
    # Let's try to generate a list first, then convert?
    # List of 14M tuples is also heavy.
    
    # Alternative: Use a set for sums to check existence, but we need coeffs.
    # Let's stick to the dict but optimize the loop.
    
    def generate_sums_map(weights_list):
        m = len(weights_list)
        # Start with sum 0, coeffs all 0
        current_sums = {0: (0,) * m}
        
        for i, w in enumerate(weights_list):
            new_sums = {}
            for s, coeffs in current_sums.items():
                # Option 1: c_i = -1
                s_new = s - w
                c_new = coeffs[:i] + (-1,) + coeffs[i+1:]
                # We only keep one coeffs per sum
                if s_new not in new_sums:
                    new_sums[s_new] = c_new
                    
                # Option 2: c_i = 0
                s_new = s
                c_new = coeffs[:i] + (0,) + coeffs[i+1:]
                if s_new not in new_sums:
                    new_sums[s_new] = c_new
                    
                # Option 3: c_i = 1
                s_new = s + w
                c_new = coeffs[:i] + (1,) + coeffs[i+1:]
                if s_new not in new_sums:
                    new_sums[s_new] = c_new
                    
            current_sums = new_sums
            # Time check
            if time.time() - start_time > deadline - 20:
                break
        return current_sums

    sums1_map = generate_sums_map(w1)
    
    if time.time() - start_time > deadline - 10:
        # Fallback
        return [1] + [0]*(n-1)

    # Sort keys for binary search
    sorted_keys1 = sorted(sums1_map.keys())
    
    # Generate sums for second half and check against first half
    # We want to minimize |s1 + s2| != 0
    min_abs_diff = float('inf')
    best_c1 = None
    best_c2 = None
    best_s1 = 0
    best_s2 = 0
    
    # Generate sums for w2 iteratively as well? 
    # Or just iterate through all possible combinations of w2?
    # 3^15 is 14M. Iterating a list of 14M items is fine.
    # Let's generate the list of (sum, coeffs) for w2.
    
    def generate_sums_list(weights_list):
        m = len(weights_list)
        current_sums = [(0, (0,) * m)]
        
        for i, w in enumerate(weights_list):
            new_sums = []
            for s, coeffs in current_sums:
                # -1
                new_sums.append((s - w, coeffs[:i] + (-1,) + coeffs[i+1:]))
                # 0
                new_sums.append((s, coeffs[:i] + (0,) + coeffs[i+1:]))
                # 1
                new_sums.append((s + w, coeffs[:i] + (1,) + coeffs[i+1:]))
            current_sums = new_sums
            if time.time() - start_time > deadline - 5:
                break
        return current_sums

    # Generating a list of 14M tuples might use ~1GB RAM.
    # Let's try to process w2 sums on the fly using DFS to save memory,
    # and for each sum, do a binary search in sorted_keys1.
    
    import bisect
    
    def dfs2(idx, current_sum, coeffs):
        nonlocal min_abs_diff, best_c1, best_c2, best_s1, best_s2
        
        if time.time() - start_time > deadline - 2:
            return
            
        if idx == len(w2):
            # Find s1 in sums1_map closest to -current_sum
            target = -current_sum
            # Binary search in sorted_keys1
            pos = bisect.bisect_left(sorted_keys1, target)
            
            # Check pos and pos-1
            candidates = []
            if pos < len(sorted_keys1):
                candidates.append(sorted_keys1[pos])
            if pos > 0:
                candidates.append(sorted_keys1[pos-1])
                
            for s1 in candidates:
                diff = abs(s1 + current_sum)
                if diff != 0 and diff < min_abs_diff:
                    min_abs_diff = diff
                    best_s1 = s1
                    best_s2 = current_sum
                    best_c1 = sums1_map[s1]
                    best_c2 = coeffs
            return
        
        w = w2[idx]
        # Try -1, 0, 1
        coeffs[idx] = -1
        dfs2(idx + 1, current_sum - w, coeffs)
        
        coeffs[idx] = 0
        dfs2(idx + 1, current_sum, coeffs)
        
        coeffs[idx] = 1
        dfs2(idx + 1, current_sum + w, coeffs)
        
    coeffs2 = [0] * len(w2)
    dfs2(0, 0, coeffs2)
    
    if best_c1 is None or best_c2 is None:
        # Fallback if something went wrong
        return [1] + [0]*(n-1)
        
    result = list(best_c1) + list(best_c2)
    
    # Verify
    total = sum(c * w for c, w in zip(result, weights))
    if total == 0:
        return [1] + [0]*(n-1)
        
    return result
