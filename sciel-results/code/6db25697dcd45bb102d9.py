import time
import sys
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    N = 10000000
    
    # 1. Compute sigma(n) for all n up to N using a fast sieve
    # sigma(n) = sum of divisors of n
    # Use the formula: sigma(n) = product over prime powers p^k || n of (p^(k+1)-1)/(p-1)
    # We'll use a sieve-based approach
    
    # Initialize sigma array
    sigma = [0] * (N + 1)
    
    # Sieve: for each i, add i to all multiples of i
    # This is O(N log N) which might be too slow for N=10^7 in Python
    # Alternative: use the multiplicative property with smallest prime factor
    
    # Let's use a linear sieve approach for sigma
    # sigma is multiplicative: if gcd(a,b)=1, sigma(ab)=sigma(a)sigma(b)
    # For prime powers: sigma(p^k) = 1 + p + p^2 + ... + p^k = (p^(k+1)-1)/(p-1)
    
    # We'll compute spf (smallest prime factor) and then compute sigma using it
    spf = list(range(N + 1))  # smallest prime factor
    for i in range(2, int(math.isqrt(N)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Now compute sigma using spf
    # sigma[n] = sigma[n / p^k] * (p^(k+1)-1)/(p-1) where p^k is the highest power of p dividing n
    sigma = [1] * (N + 1)
    # We need to track the power of the smallest prime factor
    # Let's use arrays:
    # sigma[n] - the sum of divisors
    # We'll compute iteratively
    
    # Actually, let's use a different approach:
    # For each n, find its prime factorization using spf, then compute sigma
    
    # Better: use the recurrence
    # Let n = p^k * m where p = spf(n) and p does not divide m
    # sigma(n) = sigma(m) * (1 + p + ... + p^k) = sigma(m) * (p^(k+1)-1)/(p-1)
    
    # We'll compute:
    # pow_p[n] = p^k where p^k is the highest power of spf(n) dividing n
    # sigma_part[n] = (p^(k+1)-1)/(p-1) for that p^k
    # sigma[n] = sigma[n // pow_p[n]] * sigma_part[n]
    
    pow_p = [1] * (N + 1)
    sigma_part = [1] * (N + 1)
    sigma_val = [1] * (N + 1)
    
    for n in range(2, N + 1):
        p = spf[n]
        if n % (p * p) == 0:
            # n is divisible by p^2, so pow_p[n] = pow_p[n//p] * p
            m = n // p
            pow_p[n] = pow_p[m] * p
            sigma_part[n] = sigma_part[m] + pow_p[n]
            sigma_val[n] = sigma_val[m // pow_p[m]] * sigma_part[n]
        else:
            # n is divisible by p but not p^2
            pow_p[n] = p
            sigma_part[n] = 1 + p
            m = n // p
            sigma_val[n] = sigma_val[m] * sigma_part[n]
    
    # Check time
    if time.time() > deadline - 15:
        return [11, 6]
    
    # 2. Group numbers by their sigma value
    sigma_groups = defaultdict(list)
    for n in range(1, N + 1):
        sigma_groups[sigma_val[n]].append(n)
    
    if time.time() > deadline - 8:
        return [11, 6]
    
    best_error = float('inf')
    best_a = 11
    best_b = 6
    
    # We want to minimize |a/b - 3/2| = |2a - 3b| / (2b)
    # For a given b, we want a such that 2a ≈ 3b, i.e., a ≈ 1.5b
    # And sigma(a) = sigma(b)
    
    # Iterate over b values and for each, look in the sigma group for a near 1.5*b
    # To be efficient, we can iterate b in a smart order
    
    # Sort groups by size to process smaller groups first (faster binary search)
    # Actually, let's just iterate all b and check
    
    max_b = N
    min_b = 1
    
    # For efficiency, iterate b and for each b, binary search in its sigma group
    for b in range(1, max_b + 1):
        sb = sigma_val[b]
        group = sigma_groups[sb]
        if len(group) < 2:
            continue
        
        target_a = 1.5 * b
        
        # Binary search for target_a in group
        # group is already sorted since we appended in order
        lo, hi = 0, len(group)
        while lo < hi:
            mid = (lo + hi) // 2
            if group[mid] < target_a:
                lo = mid + 1
            else:
                hi = mid
        
        # Check lo and lo-1
        candidates = []
        if lo < len(group):
            candidates.append(group[lo])
        if lo > 0:
            candidates.append(group[lo - 1])
        
        for a in candidates:
            if a == b:
                continue
            # Compute error exactly
            # |a/b - 3/2| = |2a - 3b| / (2b)
            num = abs(2 * a - 3 * b)
            if num == 0:
                # Exact match, but this is invalid per the task
                continue
            den = 2 * b
            error = num / den
            
            if error < best_error:
                best_error = error
                best_a = a
                best_b = b
        
        # Pruning: if even the best possible for this b can't beat current best, skip
        # The minimum possible error for a given b is when a is closest to 1.5b
        # But we've already checked the closest candidates
        
        # Time check every 100000 iterations
        if b % 100000 == 0:
            if time.time() > deadline - 5:
                break
    
    # Ensure types are Python ints
    a = int(best_a)
    b = int(best_b)
    
    # Final validation
    if a <= 0 or b <= 0 or a > N or b > N:
        return [11, 6]
    if sigma_val[a] != sigma_val[b]:
        return [11, 6]
    if a == b:
        return [11, 6]
    
    return [a, b]

# Test / Sanity Check if run directly
if __name__ == "__main__":
    res = construct()
    print(res)
    a, b = res
    if isinstance(a, int) and isinstance(b, int):
        print(f"a={a}, b={b}")
    else:
        print("Type error")
