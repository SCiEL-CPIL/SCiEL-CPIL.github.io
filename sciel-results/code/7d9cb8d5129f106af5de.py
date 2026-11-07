"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def _sigma(n):
    """Compute sigma(n) = sum of divisors of n, using trial division up to sqrt(n)."""
    if n == 1:
        return 1
    result = 1
    temp = n
    # Factor out 2
    count = 0
    while temp % 2 == 0:
        count += 1
        temp //= 2
    if count > 0:
        result *= (2**(count+1) - 1)
    # Odd primes
    p = 3
    while p * p <= temp:
        if temp % p == 0:
            count = 0
            power = 1
            while temp % p == 0:
                count += 1
                temp //= p
                power *= p
            result *= (power * (p + 1)) // (p - 1)
        p += 2
    if temp > 1:
        result *= (temp + 1)
    return result

def construct():
    N = 10000000
    
    # Generate candidates: squares and twice squares
    candidates = set()
    
    # Squares k^2 <= N
    max_k_sq = int(math.isqrt(N))
    for k in range(1, max_k_sq + 1):
        candidates.add(k * k)
        
    # Twice squares 2*k^2 <= N
    max_k_2sq = int(math.isqrt(N // 2))
    for k in range(1, max_k_2sq + 1):
        candidates.add(2 * k * k)
    
    # Convert to list and sort for potential use
    cand_list = sorted(candidates)
    
    # Compute sigma for each candidate and group by sigma value
    sigma_groups = {}
    for n in cand_list:
        s = _sigma(n)
        if s not in sigma_groups:
            sigma_groups[s] = []
        sigma_groups[s].append(n)
    
    best_err = float('inf')
    best_a = 1
    best_b = 1
    
    # For each group, find best pair
    for s, group in sigma_groups.items():
        if len(group) < 2:
            continue
        
        # Sort group to allow binary search
        # group is already sorted if we iterate cand_list sorted? 
        # Yes, cand_list is sorted, and we append in that order.
        
        for b in group:
            # We want a such that a/b is close to 1.5
            # a approx 1.5 * b
            target = 1.5 * b
            
            # Binary search for target in group
            # We want to check the neighbors of target in group
            # Use bisect
            import bisect
            idx = bisect.bisect_left(group, target)
            
            # Check idx and idx-1
            for i in [idx, idx - 1]:
                if 0 <= i < len(group):
                    a = group[i]
                    if a == b:
                        continue
                    # Compute error
                    # err = |a/b - 1.5| = |2a - 3b| / (2b)
                    err_num = abs(2 * a - 3 * b)
                    if err_num == 0:
                        continue
                    err = err_num / (2 * b)
                    if err < best_err:
                        best_err = err
                        best_a = a
                        best_b = b
                        
    return [best_a, best_b]
# EVOLVE-BLOCK-END
