import time
import math

# EVOLVE-BLOCK-START
MAX_N = 10000000
DEADLINE = 85.0
FALLBACK = (11, 6)

def construct():
    start = time.time()
    elapsed_limit = DEADLINE - start
    
    def check_time():
        return (time.time() - start) < elapsed_limit
    
    # Step 1: Sieve to compute sigma(n) for all n up to MAX_N
    # sigma[n] = sum of all positive divisors of n
    sigma = [0] * (MAX_N + 1)
    for i in range(1, MAX_N + 1):
        for j in range(i, MAX_N + 1, i):
            sigma[j] += i
        if not check_time():
            return [int(FALLBACK[0]), int(FALLBACK[1])]
    
    # Step 2: Group indices by sigma value
    # For each sigma value s, collect all n with sigma(n) = s
    groups = {}
    for n in range(1, MAX_N + 1):
        s = sigma[n]
        if s not in groups:
            groups[s] = []
        groups[s].append(n)
        if not check_time():
            return [int(FALLBACK[0]), int(FALLBACK[1])]
    
    # Step 3: For each group, find the pair (a, b) with a != b that minimizes |a/b - 3/2|
    # |a/b - 3/2| = |2a - 3b| / (2b)
    # We want to minimize this value.
    # For a given group, we can try all pairs, but that's O(k^2) per group.
    # Instead, for each b in the group, we want a such that a/b ≈ 3/2, i.e., a ≈ 3b/2.
    # So for each b, the best a is the element in the group closest to 3b/2.
    # We can sort the group and for each b, binary search for the closest to 3b/2.
    
    best_err = float('inf')
    best_a, best_b = FALLBACK[0], FALLBACK[1]
    
    # Sort groups by size to process larger groups first (more likely to have good pairs)
    # Actually, let's just iterate through all groups
    for s, arr in groups.items():
        if len(arr) < 2:
            continue
        arr_sorted = sorted(arr)
        # For each b in arr, find a in arr closest to 3*b/2
        # Since arr is sorted, we can use binary search
        for b in arr:
            target = 3 * b / 2.0
            # Binary search for target in arr_sorted
            lo, hi = 0, len(arr_sorted) - 1
            # Find insertion point
            while lo < hi:
                mid = (lo + hi) // 2
                if arr_sorted[mid] < target:
                    lo = mid + 1
                else:
                    hi = mid
            # Check lo and lo-1
            candidates = []
            if lo < len(arr_sorted):
                candidates.append(arr_sorted[lo])
            if lo - 1 >= 0:
                candidates.append(arr_sorted[lo - 1])
            # Also check nearby indices for safety
            for idx in range(max(0, lo - 2), min(len(arr_sorted), lo + 3)):
                if arr_sorted[idx] not in candidates:
                    candidates.append(arr_sorted[idx])
            
            for a in candidates:
                if a == b:
                    continue
                # Compute |a/b - 3/2| exactly as a rational
                # |a/b - 3/2| = |2a - 3b| / (2b)
                num = abs(2 * a - 3 * b)
                den = 2 * b
                if num == 0:
                    # This would be an exact match, but the task says raw value of exactly 0 is invalid
                    # So we skip this
                    continue
                err = num / den
                if err < best_err:
                    best_err = err
                    best_a, best_b = a, b
        
        if not check_time():
            break
    
    # Ensure we return Python ints in a list
    return [int(best_a), int(best_b)]
# EVOLVE-BLOCK-END
