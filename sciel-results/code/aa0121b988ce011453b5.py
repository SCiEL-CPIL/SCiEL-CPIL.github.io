import time
import math
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin before 90s kill
    
    n = 30
    L = 1
    for i in range(1, n + 1):
        L = L * i // math.gcd(L, i)
    
    w = [L // i for i in range(1, n + 1)]
    
    mid = n // 2
    w1 = w[:mid]
    w2 = w[mid:]
    
    # Generate all possible sums for each half
    sums1 = {0}
    for wi in w1:
        next_set = set()
        for s in sums1:
            next_set.add(s - wi)
            next_set.add(s)
            next_set.add(s + wi)
        sums1 = next_set
    
    sums2 = {0}
    for wi in w2:
        next_set = set()
        for s in sums2:
            next_set.add(s - wi)
            next_set.add(s)
            next_set.add(s + wi)
        sums2 = next_set
    
    if time.time() > deadline:
        return [0] * 28 + [1, -1]
    
    # Sort sums for binary search
    sums1_sorted = sorted(sums1)
    sums2_sorted = sorted(sums2)
    
    # Find minimum nonzero |s1 + s2| where s1 in sums1, s2 in sums2
    best_val = float('inf')
    best_s1 = None
    best_s2 = None
    
    for s2 in sums2_sorted:
        # Binary search for -s2 in sums1_sorted
        idx = bisect.bisect_left(sums1_sorted, -s2)
        
        # Check idx and idx-1
        for check_idx in [idx, idx - 1]:
            if 0 <= check_idx < len(sums1_sorted):
                s1 = sums1_sorted[check_idx]
                total = s1 + s2
                if total != 0:
                    if abs(total) < best_val:
                        best_val = abs(total)
                        best_s1 = s1
                        best_s2 = s2
                        
        # Early termination if we found |sum| = 1 (the minimum possible nonzero)
        if best_val == 1:
            break
    
    if time.time() > deadline or best_s1 is None:
        return [0] * 28 + [1, -1]
    
    # Backtrack to find coefficients for sums1
    def backtrack(weights, target_sum):
        """Find coefficients that achieve target_sum using weights."""
        coeffs = [0] * len(weights)
        current_sum = target_sum
        # Build sets incrementally for backtracking
        sets = [set([0])]
        curr = {0}
        for i, wi in enumerate(weights):
            next_set = set()
            for s in curr:
                next_set.add(s - wi)
                next_set.add(s)
                next_set.add(s + wi)
            curr = next_set
            sets.append(curr)
        
        # Backtrack from the end
        for i in range(len(weights) - 1, -1, -1):
            wi = weights[i]
            if current_sum - wi in sets[i]:
                coeffs[i] = 1
                current_sum -= wi
            elif current_sum in sets[i]:
                coeffs[i] = 0
            elif current_sum + wi in sets[i]:
                coeffs[i] = -1
                current_sum += wi
            else:
                coeffs[i] = 0
        return coeffs
    
    coeffs1 = backtrack(w1, best_s1)
    coeffs2 = backtrack(w2, best_s2)
    
    result = coeffs1 + coeffs2
    
    # Verify the result
    total = 0
    for i in range(n):
        if result[i] != 0:
            total += result[i] * L // (i + 1)
    
    # If something went wrong, return a valid nonzero result
    if total == 0:
        return [0] * 28 + [1, -1]
    
    return result
