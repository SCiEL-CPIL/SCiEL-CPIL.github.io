"""Strategy: Meet-in-the-Middle Exhaustive Search over LCM-Integer Sums."""
# EVOLVE-BLOCK-START
import time
import math
import random
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin for 90s limit
    
    n = 30
    
    # LCM of 1..30
    L = 1
    for i in range(1, n + 1):
        L = L * i // math.gcd(L, i)
    
    # Precompute weights: weight[i] = L // (i+1)
    weights = [L // (i + 1) for i in range(n)]
    
    # Split into two halves
    mid = n // 2  # 15
    
    # Generate all possible signed sums for first half (indices 0..mid-1)
    # and second half (indices mid..n-1)
    
    def generate_half_sums(start_idx, end_idx):
        """Generate all possible integer sums for coefficients at indices start_idx..end_idx-1."""
        num_items = end_idx - start_idx
        # Use iterative building
        sums = [0]
        for j in range(num_items):
            w = weights[start_idx + j]
            new_sums = []
            for s in sums:
                new_sums.append(s - w)
                new_sums.append(s)
                new_sums.append(s + w)
            sums = new_sums
        return sums
    
    # Enumerate first half
    half1_sums = generate_half_sums(0, mid)
    # Enumerate second half
    half2_sums = generate_half_sums(mid, n)
    
    # We want to find s1 + s2 != 0 minimizing |s1 + s2|
    # i.e., s2 = -s1 + delta, minimize |delta| where delta != 0
    
    # Sort half2_sums
    half2_sorted = sorted(half2_sums)
    
    best_delta = None
    best_s1 = None
    best_s2 = None
    
    # For each s1 in half1_sums, find s2 in half2_sorted closest to -s1
    # but not exactly -s1 (which would give sum=0)
    
    import bisect
    
    for s1 in half1_sums:
        target = -s1
        # Find insertion point in half2_sorted
        pos = bisect.bisect_left(half2_sorted, target)
        
        # Check pos-1 and pos (and maybe pos+1 for safety)
        candidates = []
        if pos > 0:
            candidates.append(pos - 1)
        if pos < len(half2_sorted):
            candidates.append(pos)
        if pos + 1 < len(half2_sorted):
            candidates.append(pos + 1)
        
        for p in candidates:
            s2 = half2_sorted[p]
            delta = s1 + s2
            if delta == 0:
                continue
            abs_delta = abs(delta)
            if best_delta is None or abs_delta < best_delta:
                best_delta = abs_delta
                best_s1 = s1
                best_s2 = s2
                # If we found delta=1, that's the absolute minimum possible
                if best_delta == 1:
                    break
        if best_delta == 1:
            break
    
    if best_delta is None:
        # Fallback: should not happen, but just in case
        return [0] * 28 + [1, -1]
    
    # Now we need to recover the actual coefficients
    # We need to find which combination of coefficients gives s1 and which gives s2
    
    # For half 1: find coeffs c[0..mid-1] in {-1,0,1} such that sum c[i]*weights[i] = best_s1
    # For half 2: find coeffs c[mid..n-1] in {-1,0,1} such that sum c[i]*weights[i] = best_s2
    
    def recover_coeffs(start_idx, end_idx, target_sum):
        """Recover coefficients that produce target_sum for the given index range."""
        num_items = end_idx - start_idx
        # Backtracking search
        coeffs = [0] * num_items
        
        def backtrack(idx, current_sum):
            if idx == num_items:
                return current_sum == target_sum
            w = weights[start_idx + idx]
            for val in [1, -1, 0]:
                coeffs[idx] = val
                if backtrack(idx + 1, current_sum + val * w):
                    return True
            return False
        
        if backtrack(0, 0):
            return coeffs
        return None
    
    half1_coeffs = recover_coeffs(0, mid, best_s1)
    half2_coeffs = recover_coeffs(mid, n, best_s2)
    
    if half1_coeffs is None or half2_coeffs is None:
        # Fallback
        return [0] * 28 + [1, -1]
    
    result = half1_coeffs + half2_coeffs
    
    # Verify
    total = sum(c * w for c, w in zip(result, weights))
    if total == 0:
        # Should not happen, but fix
        result[0] = 1
        total = sum(c * w for c, w in zip(result, weights))
    
    return result
# EVOLVE-BLOCK-END
