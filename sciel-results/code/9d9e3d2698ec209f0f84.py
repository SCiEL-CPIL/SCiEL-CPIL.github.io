import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    n = 30
    
    # 1. Compute L = lcm(1, ..., 30)
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
        
    # 2. Compute weights w_i = L // i
    w = [0] * (n + 1)
    for i in range(1, n + 1):
        w[i] = L // i
        
    mid = n // 2
    # Indices 1..mid for left, mid+1..n for right
    left_weights = [w[i] for i in range(1, mid + 1)]
    right_weights = [w[i] for i in range(mid + 1, n + 1)]
    
    # 3. Generate all sums and coefficient encodings for left half
    # D encodes coefficients in base 3: 0->-1, 1->0, 2->1
    # We generate a list of (sum, D)
    
    def generate_sums(weights):
        # Returns list of (sum, D)
        results = [(0, 0)]
        for idx, wt in enumerate(weights):
            power = 3 ** idx
            new_results = []
            for s, d in results:
                # c = -1 -> d_digit = 0
                new_results.append((s - wt, d))
                # c = 0 -> d_digit = 1
                new_results.append((s, d + power))
                # c = 1 -> d_digit = 2
                new_results.append((s + wt, d + 2 * power))
            results = new_results
        return results

    # This loop is $3^{15}$ iterations in the last step.
    # Total operations sum_{k=0}^{14} 3^k * 3 = 3^{15} - 1.
    # ~14 million appends.
    
    if time.time() - start_time > deadline:
        # Fallback if generation is too slow (unlikely)
        return [1] + [0]*29
        
    left_sums = generate_sums(left_weights)
    
    # Sort by sum
    # We want to binary search on the sum.
    # Sorting a list of tuples sorts by first element, then second.
    left_sums.sort(key=lambda x: x[0])
    
    # Extract sums for binary search
    # bisect works on the whole list if we use a tuple key, 
    # but we want to find the sum closest to target.
    # We can create a list of just the sums for faster bisect?
    # bisect on list of tuples is fine if we search for (target, -inf)
    
    sums_only = [x[0] for x in left_sums]
    
    import bisect
    
    best_abs = None
    best_sum_val = None
    best_D1 = None
    best_D2 = None
    
    # 4. Generate right half sums and check against left
    # We can generate on the fly to save memory?
    # generate_sums returns a list. 14 million tuples is ~1GB?
    # Let's try to generate right sums on the fly and query left_sums.
    
    # Recursive generation for right half to avoid storing 14M tuples if possible?
    # Or just generate the list, it's the same size as left.
    # Memory: 2 * 14M tuples. Might be tight.
    # Let's try to generate right sums iteratively and process them.
    
    # Actually, we can just call generate_sums(right_weights)
    # If memory fails, we can switch to a different strategy.
    # Given 60s limit, let's assume memory is sufficient (typically 2-4GB+).
    
    right_sums = generate_sums(right_weights)
    
    for s2, d2 in right_sums:
        target = -s2
        # Find index in sums_only closest to target
        idx = bisect.bisect_left(sums_only, target)
        
        # Check idx and idx-1
        candidates = []
        if idx < len(sums_only):
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)
            
        for i in candidates:
            s1, d1 = left_sums[i]
            total_sum = s1 + s2
            if total_sum == 0:
                continue
            abs_total = abs(total_sum)
            if best_abs is None or abs_total < best_abs:
                best_abs = abs_total
                best_sum_val = total_sum
                best_D1 = d1
                best_D2 = d2
                
    if best_D1 is None:
        # Fallback
        return [1] + [0]*29
        
    # 5. Decode coefficients
    coeffs = [0] * n
    
    # Decode D1 for indices 1..mid
    d = best_D1
    for i in range(mid):
        digit = d % 3
        d //= 3
        c = digit - 1 # 0->-1, 1->0, 2->1
        coeffs[i] = c
        
    # Decode D2 for indices mid+1..n
    d = best_D2
    for i in range(n - mid):
        digit = d % 3
        d //= 3
        c = digit - 1
        coeffs[mid + i] = c
        
    return coeffs
