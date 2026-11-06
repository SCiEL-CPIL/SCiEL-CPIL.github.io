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
    
    for p in primes:
        if p * p > start + length - 1:
            break
        
        # Find first multiple of p in [start, start+length-1]
        r = start % p
        if r == 0:
            first = start
        else:
            first = start + (p - r)
        
        if first > start + length - 1:
            continue
        
        for multiple in range(first, start + length, p):
            idx = multiple - start
            phi_val[idx] = phi_val[idx] // p * (p - 1)
            # rem[idx] is guaranteed to be divisible by p at this point
            # because we haven't processed p yet for this idx, and p divides multiple
            rem[idx] //= p
            while rem[idx] % p == 0:
                rem[idx] //= p
                
    for i in range(length):
        if rem[i] > 1:
            phi_val[i] = phi_val[i] // rem[i] * (rem[i] - 1)
            
    return phi_val

def find_longest_distinct_run(phis):
    """Find the longest contiguous subarray with all distinct values.
    Returns (length, start_index)."""
    n = len(phis)
    if n == 0:
        return 0, 0
    best = 1
    best_start = 0
    left = 0
    seen = {}
    for right, val in enumerate(phis):
        if val in seen and seen[val] >= left:
            left = seen[val] + 1
        seen[val] = right
        current_len = right - left + 1
        if current_len > best:
            best = current_len
            best_start = left
    return best, best_start

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin
    
    # Generate primes up to 10^6 (sufficient for n up to 10^12)
    prime_limit = 10**6
    primes = sieve_primes(prime_limit)
    
    max_k = 200
    max_n = 10**12
    
    best_k = 1
    best_start = 1
    
    # Strategy: Scan large contiguous ranges
    # The probability of distinct totients increases with n.
    # We scan ranges of size ~10^5 to 10^6.
    
    scan_size = 10**5 # Start with 100k
    # We can afford to scan maybe 10-20 such blocks in 120s?
    # Let's try a few specific large ranges.
    
    ranges_to_scan = []
    # High magnitude ranges
    for power in range(6, 13):
        lo = 10**power
        if lo + scan_size > max_n:
            continue
        # Add a few random offsets in this decade
        for _ in range(5):
            offset = random.randint(0, 10**power)
            ranges_to_scan.append((lo + offset, scan_size))
            
    # Also scan near 10^12
    for _ in range(10):
        lo = random.randint(int(10**11), max_n - scan_size)
        ranges_to_scan.append((lo, scan_size))
        
    # Shuffle to vary order
    random.shuffle(ranges_to_scan)
    
    for start, length in ranges_to_scan:
        if time.time() > deadline - 10:
            break
        # Compute phi for this segment
        phis = compute_phi_segment(start, length, primes)
        run_len, run_start = find_longest_distinct_run(phis)
        if run_len > best_k:
            best_k = run_len
            best_start = start + run_start
            if best_k == max_k:
                break
        
        # If we found a long run, try to extend it?
        # Or just continue scanning.
        
    # If we didn't find max_k, do a finer search around the best found?
    # Or random 200-length checks if time permits.
    if best_k < max_k and time.time() < deadline - 5:
        for _ in range(50):
            if time.time() > deadline - 2:
                break
            s = random.randint(1, max_n - max_k + 1)
            phis = compute_phi_segment(s, max_k, primes)
            if len(set(phis)) == max_k:
                return {'start': s, 'length': max_k}
            run_len, run_start = find_longest_distinct_run(phis)
            if run_len > best_k:
                best_k = run_len
                best_start = s + run_start

    return {'start': best_start, 'length': best_k}

# EVOLVE-BLOCK-END
