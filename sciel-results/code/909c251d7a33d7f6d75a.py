"""Optimized program to find long runs of consecutive integers with distinct totients."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def sieve_primes(limit):
    """Generate all primes up to limit using Sieve of Eratosthenes."""
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(limit + 1) if is_prime[i]]

def segmented_spf(start, length, primes):
    """
    Compute smallest prime factor for each number in [start, start+length-1].
    Returns an array spf where spf[i] is the smallest prime factor of start+i,
    or 0 if the number is prime (or 1).
    """
    n = start + length - 1
    spf = np.zeros(length, dtype=np.int64)
    
    # For each prime, mark its multiples in the range
    for p in primes:
        if p > n:
            break
        # Find the first multiple of p in [start, start+length-1]
        first = ((start + p - 1) // p) * p
        if first < start:
            first += p
        for j in range(first, start + length, p):
            idx = j - start
            if spf[idx] == 0:
                spf[idx] = p
    
    # Numbers that are prime (or 1) have spf = 0
    # We'll handle them separately in phi computation
    return spf

def phi_from_spf(n, spf, primes):
    """
    Compute phi(n) given that spf is the SPF array for a window starting at some base.
    Here n is the index into the spf array (0-based).
    Actually, let's compute phi for all numbers in the window at once.
    """
    pass

def phi_window(start, length, primes):
    """
    Compute phi for all numbers in [start, start+length-1] using segmented SPF.
    Returns an array of length `length` with phi values.
    """
    # Get SPF for the window
    spf = segmented_spf(start, length, primes)
    
    # Initialize phi values
    phi = np.arange(start, start + length, dtype=np.int64)
    
    # For each prime, update phi for all multiples
    for p in primes:
        if p > start + length - 1:
            break
        first = ((start + p - 1) // p) * p
        if first < start:
            first += p
        for j in range(first, start + length, p):
            idx = j - start
            phi[idx] = phi[idx] // p * (p - 1)
    
    return phi

def find_max_distinct_run(phi_vals):
    """
    Given an array of phi values, find the longest contiguous subarray
    with all distinct values. Returns (start_index, length).
    """
    n = len(phi_vals)
    best_len = 0
    best_start = 0
    left = 0
    seen = {}
    
    for right in range(n):
        val = phi_vals[right]
        if val in seen and seen[val] >= left:
            left = seen[val] + 1
        seen[val] = right
        if right - left + 1 > best_len:
            best_len = right - left + 1
            best_start = left
    
    return best_start, best_len

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Generate primes up to 10^6 for factoring numbers up to 10^12
    prime_limit = 10**6
    primes = sieve_primes(prime_limit)
    
    best_result = {'start': 1, 'length': 1}
    best_k_found = 1
    
    # Strategy: Use segmented phi computation to scan large windows quickly.
    # For a window of size W, we can compute phi for all W numbers, then
    # find the longest distinct run in O(W) time using sliding window.
    
    # Try windows of different sizes at various positions
    # Start with large windows near 10^12 where numbers are "random-like"
    
    window_size = 100000  # 100k numbers per window
    
    # Candidate starting positions for windows
    candidates = []
    
    # Near the top boundary
    for offset in range(0, 50000, 1000):
        s = 10**12 - 200 - offset
        if s >= 1:
            candidates.append(s)
    
    # Spread across the range
    for frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        s = int(10**12 * frac)
        for delta in range(-50000, 50001, 10000):
            s2 = s + delta
            if 1 <= s2 <= 10**12 - window_size:
                candidates.append(s2)
    
    # Pseudo-random candidates
    import random
    rng = random.Random(12345)
    for _ in range(50):
        s = rng.randint(1, 10**12 - window_size)
        candidates.append(s)
    
    candidates = list(set(candidates))
    candidates.sort(reverse=True)
    
    for s in candidates:
        if time.time() > deadline - 15:
            break
        
        if s + window_size - 1 > 10**12:
            continue
        
        # Compute phi for the entire window
        phi_vals = phi_window(s, window_size, primes)
        
        # Find the longest distinct run
        run_start, run_len = find_max_distinct_run(phi_vals)
        
        if run_len > best_k_found:
            best_k_found = run_len
            best_result = {'start': s + run_start, 'length': run_len}
            
            if run_len >= 200:
                return {'start': s + run_start, 'length': 200}
    
    # If we haven't found a long enough run, try smaller windows with more candidates
    if best_k_found < 200:
        window_size = 50000
        for s in candidates:
            if time.time() > deadline - 10:
                break
            
            if s + window_size - 1 > 10**12:
                continue
            
            phi_vals = phi_window(s, window_size, primes)
            run_start, run_len = find_max_distinct_run(phi_vals)
            
            if run_len > best_k_found:
                best_k_found = run_len
                best_result = {'start': s + run_start, 'length': run_len}
                
                if run_len >= 200:
                    return {'start': s + run_start, 'length': 200}
    
    return best_result

# EVOLVE-BLOCK-END
