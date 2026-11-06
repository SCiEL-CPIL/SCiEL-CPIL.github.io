"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def sieve_primes(limit):
    """Generate all primes up to limit using Sieve of Eratosthenes."""
    is_prime = bytearray([1]) * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i:limit+1:i] = bytearray(len(is_prime[i*i:limit+1:i]))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def compute_phi_segment(start, length, primes):
    """
    Compute phi(n) for n in [start, start+length-1] using a segmented sieve.
    Returns a list of phi values.
    """
    # Initialize phi array with n values
    # We use a list of integers for the current phi value
    # phi[n] starts as n, then we multiply by (1 - 1/p) for each prime factor p
    # It is more efficient to store the current value and update it.
    
    # We store phi_val[i] = phi(start + i)
    # Initially phi_val[i] = start + i
    phi_val = [start + i for i in range(length)]
    
    # For each prime p, find the first multiple of p in the range
    # and update phi_val for all multiples
    for p in primes:
        # Find the first multiple of p >= start
        # first = ceil(start / p) * p
        if start % p == 0:
            first = start
        else:
            first = start + (p - start % p)
        
        # If the first multiple is out of range, continue
        if first > start + length - 1:
            continue
            
        # Update phi_val for all multiples of p in the range
        # index = (multiple - start)
        for multiple in range(first, start + length, p):
            idx = multiple - start
            # phi(n) = n * (1 - 1/p) = n * (p-1) / p
            # We can update as: phi_val[idx] -= phi_val[idx] // p
            # This works because phi_val[idx] is divisible by p at this stage?
            # Wait, phi_val[idx] is initially multiple.
            # If we update phi_val[idx] -= phi_val[idx] // p, this is correct for the first factor.
            # However, if a number has p^2, we only apply the factor (1-1/p) once.
            # The standard segmented sieve for phi updates each prime factor once.
            # So this logic holds.
            phi_val[idx] -= phi_val[idx] // p
            
    return phi_val

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Generate primes up to 10^6
    prime_limit = 10**6
    primes = sieve_primes(prime_limit)
    
    max_k = 200
    max_n = 10**12
    
    # We want to find s such that phi(s)..phi(s+max_k-1) are distinct.
    # Strategy: Scan through the range [1, max_n] in steps.
    # Given the speed of compute_phi_segment, we can scan a lot.
    # Let's try scanning from high values down, or random spots.
    # High values are "safer" in terms of phi(n) being large and distinct.
    
    # Let's try a deterministic scan near the upper bound first.
    # If we find a valid run of length 200, we return it.
    
    # To be efficient, we can check multiple segments in parallel or just sequentially.
    # Sequential is fine if each check is ~microseconds.
    
    # Let's try starting points.
    # We'll check segments of length 200.
    # If we find one, we return it.
    
    # To avoid missing a valid segment, we can step by 1 or by a small amount.
    # Let's step by 1 for a while, then increase step or jump.
    
    # Since the probability of collision is very low, almost any segment should work.
    # Let's verify with a few random segments first to ensure no bugs.
    
    random.seed(42)
    
    # Try some random segments first
    for _ in range(100):
        if time.time() > deadline:
            break
        s = random.randint(1, max_n - max_k + 1)
        phis = compute_phi_segment(s, max_k, primes)
        if len(set(phis)) == max_k:
            return {'start': s, 'length': max_k}
            
    # If random didn't find it (unlikely), do a structured search
    # Start from max_n - max_k + 1 and go down
    s = max_n - max_k + 1
    step = 1
    
    while s >= 1:
        if time.time() > deadline:
            break
        
        phis = compute_phi_segment(s, max_k, primes)
        if len(set(phis)) == max_k:
            return {'start': s, 'length': max_k}
        
        s -= step
        
        # Optionally increase step if we are not finding any
        # But since they should be very common, step 1 should be fine.
        
    # Fallback: Return a smaller valid run if we ran out of time or space
    # This shouldn't happen, but good for safety.
    # Let's try to find the longest valid run from a fixed point.
    s = 10**12 - 1000
    for k in range(max_k, 1, -1):
        if s + k - 1 > max_n:
            continue
        phis = compute_phi_segment(s, k, primes)
        if len(set(phis)) == k:
            return {'start': s, 'length': k}
            
    # Absolute fallback
    return {'start': 1, 'length': 1}

# EVOLVE-BLOCK-END
