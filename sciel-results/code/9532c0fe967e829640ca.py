import time
import math
import random

def sieve_primes(limit):
    """Generate all primes up to limit using Sieve of Eratosthenes."""
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i:limit+1:i] = bytearray(len(is_prime[i*i:limit+1:i]))
    return [i for i, v in enumerate(is_prime) if v]

def compute_phi_window(L, W, primes):
    """Compute phi(n) for n in [L, L+W-1] using precomputed primes."""
    phi = list(range(W))
    for p in primes:
        if p > W and p > L:
            break
        # Find the first multiple of p in [L, L+W-1]
        start = ((L + p - 1) // p) * p
        for m in range(start, L + W, p):
            idx = m - L
            if phi[idx] % p == 0:
                phi[idx] -= phi[idx] // p
    return phi

def construct():
    start_time = time.time()
    deadline = start_time + 118
    
    # Precompute primes up to 10^6
    prime_limit = 10**6
    primes = sieve_primes(prime_limit)
    
    max_k = 200
    max_n = 10**12
    window_size = 10**6
    
    best_k = 1
    best_s = 1
    
    # Sample random windows
    num_samples = 100
    for _ in range(num_samples):
        if time.time() > deadline:
            break
        
        # Choose a random start position
        L = random.randint(1, max_n - window_size)
        
        # Compute phi for the window
        phi_values = compute_phi_window(L, window_size, primes)
        
        # Find the longest run of distinct phi values
        seen = set()
        current_start = 0
        for i in range(window_size):
            if phi_values[i] in seen:
                # Check if the current run is the best
                current_len = i - current_start
                if current_len > best_k:
                    best_k = current_len
                    best_s = L + current_start
                # Reset the window
                seen = set()
                current_start = i
            seen.add(phi_values[i])
        
        # Check the last run
        current_len = window_size - current_start
        if current_len > best_k:
            best_k = current_len
            best_s = L + current_start
    
    return {'start': best_s, 'length': best_k}
