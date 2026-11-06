"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def sieve_primes(limit):
    """Generate all primes up to limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            step = i
            start = i * i
            is_prime[start:limit+1:step] = bytearray(len(is_prime[start:limit+1:step]))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def compute_phi_segment(start, length, primes):
    """
    Compute exact phi(n) for n in [start, start+length-1].
    Returns list of phi values.
    """
    phi_val = list(range(start, start + length))
    rem = list(range(start, start + length))
    
    max_n = start + length - 1
    sqrt_max = int(math.isqrt(max_n)) + 1
    
    for p in primes:
        if p > sqrt_max:
            break
        
        # Find first multiple of p in [start, start+length-1]
        r = start % p
        if r == 0:
            first = start
        else:
            first = start + (p - r)
        
        if first > max_n:
            continue
        
        for multiple in range(first, start + length, p):
            idx = multiple - start
            phi_val[idx] = phi_val[idx] // p * (p - 1)
            while rem[idx] % p == 0:
                rem[idx] //= p
                
    for i in range(length):
        if rem[i] > 1:
            phi_val[i] = phi_val[i] // rem[i] * (rem[i] - 1)
            
    return phi_val

def find_longest_distinct_run(phis):
    """
    Find the longest contiguous subarray of phis with all distinct values.
    Uses sliding window with hash set for O(n) time.
    Returns (start_index, length) of the longest such subarray.
    """
    n = len(phis)
    if n == 0:
        return (0, 0)
    
    best_start = 0
    best_len = 1
    seen = {}
    left = 0
    
    for right in range(n):
        val = phis[right]
        if val in seen and seen[val] >= left:
            left = seen[val] + 1
        seen[val] = right
        
        if right - left + 1 > best_len:
            best_len = right - left + 1
            best_start = left
    
    return (best_start, best_len)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    # Generate primes up to 10^6 (sufficient for n up to 10^12)
    prime_limit = 10**6
    primes = sieve_primes(prime_limit)
    
    max_k = 200
    max_n = 10**12
    
    best_k = 1
    best_start = 1
    
    # STRATEGY 1: Deterministic candidates near 10^12
    # For large s, collision probability is ~k^2/s which is tiny.
    base = max_n - max_k + 1  # = 999999999801, so base + 199 = 10^12
    
    deterministic_starts = []
    # Try well-spaced offsets for robustness
    for off in [0, 100, 200, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]:
        s = base - off
        if s >= 1:
            deterministic_starts.append(s)
    
    # Also try some mid-range deterministic points
    deterministic_starts.extend([
        999999900000 - max_k + 1,
        999900000000 - max_k + 1,
        999000000000 - max_k + 1,
        990000000000 - max_k + 1,
        900000000000 - max_k + 1,
        500000000000 - max_k + 1,
        100000000000 - max_k + 1,
    ])
    
    # Deduplicate while preserving order
    seen_starts = set()
    unique_starts = []
    for s in deterministic_starts:
        if s not in seen_starts and s >= 1 and s + max_k - 1 <= max_n:
            seen_starts.add(s)
            unique_starts.append(s)
    
    for s in unique_starts:
        if time.time() > deadline - 30:
            break
        phis = compute_phi_segment(s, max_k, primes)
        if len(set(phis)) == max_k:
            return {'start': s, 'length': max_k}
        run_start, run_len = find_longest_distinct_run(phis)
        if run_len > best_k:
            best_k = run_len
            best_start = s + run_start
    
    # STRATEGY 2: Random search in high range with more samples
    random.seed(42)
    for _ in range(10000):
        if time.time() > deadline - 10:
            break
        s = random.randint(max(1, max_n - 10**9), max_n - max_k + 1)
        phis = compute_phi_segment(s, max_k, primes)
        if len(set(phis)) == max_k:
            return {'start': s, 'length': max_k}
        run_start, run_len = find_longest_distinct_run(phis)
        if run_len > best_k:
            best_k = run_len
            best_start = s + run_start
    
    # STRATEGY 3: Random search across all ranges
    for _ in range(2000):
        if time.time() > deadline - 5:
            break
        s = random.randint(1, max_n - max_k + 1)
        phis = compute_phi_segment(s, max_k, primes)
        if len(set(phis)) == max_k:
            return {'start': s, 'length': max_k}
        run_start, run_len = find_longest_distinct_run(phis)
        if run_len > best_k:
            best_k = run_len
            best_start = s + run_start
    
    # STRATEGY 4: Local refinement around best found
    if best_k >= 50 and time.time() < deadline - 3:
        for delta in range(-200, 201, 1):
            if time.time() > deadline - 2:
                break
            s = best_start + delta
            if s < 1 or s + max_k - 1 > max_n:
                continue
            phis = compute_phi_segment(s, max_k, primes)
            if len(set(phis)) == max_k:
                return {'start': s, 'length': max_k}
            run_start, run_len = find_longest_distinct_run(phis)
            if run_len > best_k:
                best_k = run_len
                best_start = s + run_start
    
    return {'start': best_start, 'length': best_k}

# EVOLVE-BLOCK-END
