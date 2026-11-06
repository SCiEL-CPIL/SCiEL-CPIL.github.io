import time
import math
import numpy as np
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    n = 30
    L = 1
    for i in range(1, n + 1):
        L = L * i // math.gcd(L, i)
        
    w = [L // i for i in range(1, n + 1)]
    
    mid = n // 2
    w1 = np.array(w[:mid], dtype=np.int64)
    w2 = np.array(w[mid:], dtype=np.int64)
    
    def generate_sums_numpy(weights):
        """Generate all possible signed sums using NumPy vectorization."""
        # Start with [0]
        sums = np.array([0], dtype=np.int64)
        for wi in weights:
            # For each existing sum, create three new sums: s-wi, s, s+wi
            new_sums = np.concatenate([sums - wi, sums, sums + wi])
            sums = new_sums
            if time.time() > deadline:
                return None
        return sums
    
    def generate_sums_with_history(weights):
        """Generate all sums and store history for backtracking."""
        history = [set([0])]
        curr = {0}
        for wi in weights:
            next_set = set()
            for s in curr:
                next_set.add(s - wi)
                next_set.add(s)
                next_set.add(s + wi)
            curr = next_set
            history.append(curr)
        return list(curr), history
    
    def backtrack_coeffs(weights, history, target_sum):
        """Backtrack to find coefficients that achieve target_sum."""
        coeffs = [0] * len(weights)
        current_sum = target_sum
        for i in range(len(weights) - 1, -1, -1):
            wi = weights[i]
            if current_sum - wi in history[i]:
                coeffs[i] = 1
                current_sum -= wi
            elif current_sum in history[i]:
                coeffs[i] = 0
            elif current_sum + wi in history[i]:
                coeffs[i] = -1
                current_sum += wi
            else:
                coeffs[i] = 0
        return coeffs
    
    # Generate sums using NumPy for speed
    sums1 = generate_sums_numpy(w1)
    if sums1 is None:
        return [0] * 28 + [1, -1]
        
    sums2 = generate_sums_numpy(w2)
    if sums2 is None:
        return [0] * 28 + [1, -1]
    
    if time.time() > deadline:
        return [0] * 28 + [1, -1]
    
    # Sort sums1 for binary search
    sums1_sorted = np.sort(sums1)
    
    best_val = float('inf')
    best_s1 = None
    best_s2 = None
    
    # For each sum in sums2, binary search in sums1 for closest to -s2
    # Vectorize where possible
    for s2 in sums2:
        target = -s2
        idx = bisect.bisect_left(sums1_sorted, target)
        
        # Check idx and idx-1
        for check_idx in [idx, idx - 1]:
            if 0 <= check_idx < len(sums1_sorted):
                s1 = sums1_sorted[check_idx]
                total = s1 + s2
                if total != 0:
                    abs_total = abs(total)
                    if abs_total < best_val:
                        best_val = abs_total
                        best_s1 = s1
                        best_s2 = s2
    
    if time.time() > deadline or best_s1 is None:
        return [0] * 28 + [1, -1]
    
    # Now we need to recover the coefficients.
    # Use history-based backtracking
    sums1_list, hist1 = generate_sums_with_history(w[:mid])
    sums2_list, hist2 = generate_sums_with_history(w[mid:])
    
    coeffs1 = backtrack_coeffs(w[:mid], hist1, best_s1)
    coeffs2 = backtrack_coeffs(w[mid:], hist2, best_s2)
    
    return coeffs1 + coeffs2
