import time
import random
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    n = 30
    
    # Compute LCM(1..30)
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
    
    # Scaled weights: w[i] = L // i for i in 1..n
    w = [L // i for i in range(1, n + 1)]
    
    # Meet-in-the-middle: split into two halves of 15
    half = n // 2
    left_w = w[:half]
    right_w = w[half:]
    
    # Enumerate all 3^15 possibilities for left half
    # Each coefficient is -1, 0, or 1
    # Sum = sum(c_i * w_i)
    
    def enumerate_sums(weights):
        """Enumerate all possible sums for given weights with coeffs in {-1,0,1}."""
        sums = [0]
        for w_i in weights:
            new_sums = []
            for s in sums:
                new_sums.append(s)       # c_i = 0
                new_sums.append(s + w_i) # c_i = 1
                new_sums.append(s - w_i) # c_i = -1
            sums = new_sums
        return sums
    
    # Check time budget
    if (time.time() - start_time) > deadline - 30:
        # Not enough time for MITM, fall back to simple approach
        return [0] * 28 + [1, -1]
    
    # Enumerate left sums
    left_sums = enumerate_sums(left_w)
    
    if (time.time() - start_time) > deadline - 15:
        return [0] * 28 + [1, -1]
    
    # Enumerate right sums
    right_sums = enumerate_sums(right_w)
    
    if (time.time() - start_time) > deadline - 5:
        return [0] * 28 + [1, -1]
    
    # Now find the minimum |N| > 0 where N = left_sum + right_sum
    # Sort right_sums for binary search
    right_sorted = sorted(right_sums)
    right_set = set(right_sums)
    
    best_abs_N = None
    best_left_idx = None
    best_right_val = None
    
    # For each left sum, find the right sum that minimizes |left + right|
    # We want left + right != 0, and |left + right| minimal
    
    # Use two-pointer or binary search approach
    # For each left_sum, we want right_sum closest to -left_sum but not equal to -left_sum
    
    import bisect
    
    for ls in left_sums:
        target = -ls
        # Find position in right_sorted where target would be
        pos = bisect.bisect_left(right_sorted, target)
        
        # Check candidates around pos
        candidates = []
        if pos < len(right_sorted):
            candidates.append(right_sorted[pos])
        if pos > 0:
            candidates.append(right_sorted[pos - 1])
        if pos + 1 < len(right_sorted):
            candidates.append(right_sorted[pos + 1])
        if pos - 2 >= 0:
            candidates.append(right_sorted[pos - 2])
        
        for rs in candidates:
            total = ls + rs
            if total == 0:
                continue
            abs_total = abs(total)
            if best_abs_N is None or abs_total < best_abs_N:
                best_abs_N = abs_total
                best_left_idx = left_sums.index(ls)  # This is O(n), let's optimize
                best_right_val = rs
    
    # Actually, let's be more careful. We need to recover the coefficients.
    # Let me redo this more carefully with coefficient tracking.
    
    # Better approach: enumerate with coefficient vectors
    def enumerate_with_coeffs(weights):
        """Return list of (sum, coeffs) tuples."""
        results = [(0, [0] * len(weights))]
        for i, w_i in enumerate(weights):
            new_results = []
            for s, coeffs in results:
                # c_i = 0
                new_coeffs0 = coeffs.copy()
                new_results.append((s, new_coeffs0))
                # c_i = 1
                new_coeffs1 = coeffs.copy()
                new_coeffs1[i] = 1
                new_results.append((s + w_i, new_coeffs1))
                # c_i = -1
                new_coeffs_m1 = coeffs.copy()
                new_coeffs_m1[i] = -1
                new_results.append((s - w_i, new_coeffs_m1))
            results = new_results
        return results
    
    # This uses too much memory for 3^15 with full coefficient vectors.
    # Let's use a different approach: store just the sums, then reconstruct.
    
    # Actually, 3^15 = 14,348,907. Storing 14M integers is ~112MB which is fine.
    # But storing coefficient vectors is too much.
    
    # Strategy: Store sums, find best pair, then reconstruct coefficients.
    # To reconstruct, we can do a second pass.
    
    # For now, let's just find the minimum |N| and then reconstruct.
    
    # Re-enumerate but this time track which coefficient vector produces each sum
    # Actually, multiple coefficient vectors can produce the same sum.
    # We just need ANY valid coefficient vector.
    
    # Let's use a hash map: sum -> one coefficient vector that produces it
    def enumerate_sum_to_coeffs(weights):
        """Map each achievable sum to one coefficient vector that achieves it."""
        # Use dict: sum -> coeffs
        # Start with {0: [0]*len(weights)}
        # But this will have at most 3^k entries for k weights
        sum_to_coeffs = {0: [0] * len(weights)}
        
        for i, w_i in enumerate(weights):
            new_map = {}
            for s, coeffs in sum_to_coeffs.items():
                # c_i = 0
                if s not in new_map:
                    c0 = coeffs.copy()
                    new_map[s] = c0
                # c_i = 1
                s1 = s + w_i
                if s1 not in new_map:
                    c1 = coeffs.copy()
                    c1[i] = 1
                    new_map[s1] = c1
                # c_i = -1
                sm1 = s - w_i
                if sm1 not in new_map:
                    cm1 = coeffs.copy()
                    cm1[i] = -1
                    new_map[sm1] = cm1
            sum_to_coeffs = new_map
        
        return sum_to_coeffs
    
    # This is memory intensive. 3^15 entries, each with a list of 15 ints.
    # Let's estimate: 14M * (8 bytes for sum + 15*4 bytes for coeffs + overhead)
    # That's way too much.
    
    # Alternative: Just store sums, find the best, then do a targeted search.
    
    # Let me try a simpler approach: use the sorted right_sums and binary search.
    # We already have left_sums and right_sorted.
    
    # Find best pair
    best_abs_N = None
    best_ls = None
    best_rs = None
    
    for ls in left_sums:
        target = -ls
        pos = bisect.bisect_left(right_sorted, target)
        
        for offset in range(-3, 4):
            idx = pos + offset
            if 0 <= idx < len(right_sorted):
                rs = right_sorted[idx]
                total = ls + rs
                if total == 0:
                    continue
                abs_total = abs(total)
                if best_abs_N is None or abs_total < best_abs_N:
                    best_abs_N = abs_total
                    best_ls = ls
                    best_rs = rs
    
    if best_abs_N is None:
        return [0] * 28 + [1, -1]
    
    # Now reconstruct coefficients
    # We need to find coeffs_left such that sum(c_i * left_w[i]) = best_ls
    # and coeffs_right such that sum(c_i * right_w[i]) = best_rs
    
    def find_coeffs(weights, target):
        """Find coefficients in {-1,0,1} such that sum(c_i * w_i) = target."""
        n_w = len(weights)
        
        # Backtracking with pruning
        def backtrack(idx, current_sum, coeffs):
            if idx == n_w:
                if current_sum == target:
                    return coeffs.copy()
                return None
            
            remaining_w = weights[idx:]
            max_possible = sum(abs(w) for w in remaining_w)
            
            # Pruning: if |target - current_sum| > max_possible, no solution
            if abs(target - current_sum) > max_possible:
                return None
            
            for c in (0, 1, -1):
                coeffs[idx] = c
                result = backtrack(idx + 1, current_sum + c * weights[idx], coeffs)
                if result is not None:
                    return result
            
            return None
        
        return backtrack(0, 0, [0] * n_w)
    
    coeffs_left = find_coeffs(left_w, best_ls)
    coeffs_right = find_coeffs(right_w, best_rs)
    
    if coeffs_left is None or coeffs_right is None:
        # Fallback
        return [0] * 28 + [1, -1]
    
    full_coeffs = coeffs_left + coeffs_right
    
    # Verify
    total = sum(c * w[i] for i, c in enumerate(full_coeffs))
    if total == 0:
        return [0] * 28 + [1, -1]
    
    return full_coeffs
