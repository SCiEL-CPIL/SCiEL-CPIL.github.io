"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    import math
    from collections import defaultdict
    from fractions import Fraction
    
    # Meet-in-the-middle approach:
    # 1. Generate all "small" numbers and their sigma values
    # 2. For each sigma value, we have a set of numbers
    # 3. We want pairs (a, b) with sigma(a) = sigma(b) and a/b close to 3/2
    
    # First, precompute sigma for numbers up to a bound using sieve
    MAX_N = 5000000
    
    sigma = [0] * (MAX_N + 1)
    for i in range(1, MAX_N + 1):
        for j in range(i, MAX_N + 1, i):
            sigma[j] += i
    
    # Group numbers by sigma value
    sigma_to_numbers = defaultdict(list)
    for n in range(1, MAX_N + 1):
        sigma_to_numbers[sigma[n]].append(n)
    
    best_pair = None
    best_error = Fraction(1, 1)  # Use exact rational arithmetic
    
    # For each sigma value with multiple numbers, find pairs with ratio close to 3/2
    for sig_val, nums in sigma_to_numbers.items():
        if len(nums) < 2:
            continue
        nums.sort()
        n_len = len(nums)
        
        # For each b, find a such that a/b ≈ 3/2
        # a ≈ 3b/2
        for i in range(n_len):
            b = nums[i]
            target_a = (3 * b) // 2
            
            # Binary search for target_a in sorted nums
            lo, hi = 0, n_len - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if nums[mid] < target_a:
                    lo = mid + 1
                else:
                    hi = mid
            
            # Check lo and lo-1 (and maybe lo+1 for safety)
            for idx in [lo - 1, lo, lo + 1]:
                if 0 <= idx < n_len:
                    a = nums[idx]
                    if a == b:
                        continue
                    if a > 10000000 or b > 10000000:
                        continue
                    # Exact rational error: |a/b - 3/2| = |2a - 3b| / (2b)
                    err_num = abs(2 * a - 3 * b)
                    err_den = 2 * b
                    if err_num == 0:
                        continue  # Exact 3/2 ratio is invalid
                    err = Fraction(err_num, err_den)
                    if err < best_error:
                        best_error = err
                        best_pair = (a, b)
    
    if best_pair is None:
        return [122, 80]
    
    a, b = best_pair
    if a <= 0 or b <= 0 or a > 10000000 or b > 10000000:
        return [122, 80]
    
    return [int(a), int(b)]

# EVOLVE-BLOCK-END
