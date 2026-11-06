import time
import math
import random
import numpy as np

def sieve_primes(limit):
    """Generate all primes up to limit using Sieve of Eratosthenes."""
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i:limit+1:i] = bytearray(len(is_prime[i*i:limit+1:i]))
    return [i for i, v in enumerate(is_prime) if v]

def phi_segment(start, length, primes):
    """Compute phi(n) for n in [start, start+length-1] using segmented sieve."""
    phi_vals = list(range(start, start + length))
    remaining = list(range(start, start + length))
    
    max_val = start + length - 1
    sqrt_max = int(math.isqrt(max_val))
    
    for p in primes:
        if p > sqrt_max:
            break
        first = ((start + p - 1) // p) * p
        for m in range(first, start + length, p):
            idx = m - start
            phi_vals[idx] -= phi_vals[idx] // p
            while remaining[idx] % p == 0:
                remaining[idx] //= p
    
    for i in range(length):
        if remaining[i] > 1:
            phi_vals[i] -= phi_vals[i] // remaining[i]
    
    return phi_vals

def check_run(phi_vals, start_idx, k):
    """Check if phi_vals[start_idx:start_idx+k] are all distinct."""
    sub = phi_vals[start_idx:start_idx+k]
    return len(set(sub)) == k

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    # Precompute primes up to 10^6
    prime_limit = 10**6
    primes = sieve_primes(prime_limit)
    
    best_result = {'start': 4, 'length': 2}
    
    # Strategy: For a given window size W, compute phi for [s, s+W-1],
    # then check all length-k sub-windows. This amortizes the phi computation.
    
    # We'll try different window sizes and search positions.
    # Near 10^12, collisions are rare, so even k=200 should be findable.
    
    # First, try to find k=200 near 10^12
    # Use a window of size 250 to allow some slack
    window_size = 250
    
    search_regions = []
    
    # Near 10^12 (highest priority - collisions rarest)
    for offset in range(0, 5000, 1):
        search_regions.append(10**12 - 199 - offset)
    
    # Also try some regions in the middle
    random.seed(12345)
    for _ in range(20000):
        s = random.randint(1, 10**12 - 199)
        search_regions.append(s)
    
    # Small values
    for s in range(1, 5000):
        search_regions.append(s)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_regions = []
    for s in search_regions:
        if s >= 1 and s + window_size - 1 <= 10**12 and s not in seen:
            seen.add(s)
            unique_regions.append(s)
    
    # Try each region with window_size, checking for k=200 first, then smaller
    max_k_to_try = 200
    
    for s in unique_regions:
        if time.time() > deadline - 3:
            break
        
        # Compute phi for the window
        phi_vals = phi_segment(s, window_size, primes)
        
        # Check for k=200
        if check_run(phi_vals, 0, 200):
            return {'start': s, 'length': 200}
        
        # Check for k=199, 198, ... down to some minimum
        for k in range(199, 50, -1):
            if k > window_size:
                continue
            # Check all sub-windows of length k
            found = False
            for offset in range(0, window_size - k + 1):
                if check_run(phi_vals, offset, k):
                    if k > best_result['length']:
                        best_result = {'start': s + offset, 'length': k}
                    found = True
                    break
            if found:
                break
    
    # If we haven't found k=200, try a more targeted search
    # Use a larger window and check more carefully near 10^12
    if best_result['length'] < 200:
        large_window = 400
        for offset in range(0, 10000):
            if time.time() > deadline - 3:
                break
            s = 10**12 - 199 - offset
            if s < 1:
                break
            phi_vals = phi_segment(s, large_window, primes)
            
            for k in range(200, 100, -1):
                for off in range(0, large_window - k + 1):
                    if check_run(phi_vals, off, k):
                        if k > best_result['length']:
                            best_result = {'start': s + off, 'length': k}
                        break
                if best_result['length'] >= k:
                    break
    
    return best_result
