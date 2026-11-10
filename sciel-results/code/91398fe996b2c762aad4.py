import math
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s
    
    # 1. Generate primes up to 10^6
    limit = 10**6
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    if limit >= 1:
        is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    # We want to find a run of length 200.
    # We search in a high range to minimize collision probability.
    # Let's start at 10^11.
    
    target_k = 200
    start_base = 10**11
    
    # Function to compute phi for a segment [s, s + k - 1]
    def compute_phi_segment(s, k):
        # phi_vals[i] corresponds to phi(s+i)
        phi_vals = [0] * k
        remaining = [0] * k
        for i in range(k):
            val = s + i
            phi_vals[i] = val
            remaining[i] = val
        
        # Process each prime
        # We only need primes up to sqrt(s + k - 1)
        max_val = s + k - 1
        sqrt_max = int(math.isqrt(max_val))
        
        for p in primes:
            if p > sqrt_max:
                break
            
            # Find first multiple of p in [s, s+k-1]
            # first_multiple = ceil(s / p) * p
            if s % p == 0:
                first_multiple = s
            else:
                first_multiple = ((s // p) + 1) * p
            
            # Iterate through multiples
            for m in range(first_multiple, s + k, p):
                idx = m - s
                if remaining[idx] % p == 0:
                    # p is a factor
                    phi_vals[idx] = phi_vals[idx] // p * (p - 1)
                    while remaining[idx] % p == 0:
                        remaining[idx] //= p
        
        # Handle remaining prime factors
        for i in range(k):
            if remaining[i] > 1:
                p = remaining[i]
                phi_vals[i] = phi_vals[i] // p * (p - 1)
        
        return phi_vals

    # Search for a valid start
    # We will try a few starting points just in case, but one should suffice.
    # We can step by 1, or maybe larger steps if we want to be faster, 
    # but 1 is fine since the check is cheap.
    
    current_s = start_base
    
    # To avoid getting stuck if a specific range is bad (unlikely), 
    # we can just keep going.
    
    while True:
        if time.time() > deadline:
            # Fallback: return a known valid small range or just a single element?
            # The task requires maximizing k. If we time out, we should return 
            # the best found so far. But we expect to find 200 immediately.
            # For safety, return a valid small result if we haven't found 200.
            # But the prompt implies we should find 200.
            # Let's just return a single element to be valid, though suboptimal.
            # Actually, we can cache the best found.
            break
            
        phi_vals = compute_phi_segment(current_s, target_k)
        
        # Check for distinctness
        if len(set(phi_vals)) == target_k:
            return {'start': current_s, 'length': target_k}
        
        current_s += 1

    # If loop broke due to time, return a valid (though suboptimal) result
    # We can find a small run quickly.
    # Or just return start=1, length=1 which is always valid.
    # But let's try to find a small run quickly in the fallback.
    # Actually, the probability of not finding 200 at 10^11 is negligible.
    # We'll return a single element to satisfy the interface if we fail.
    return {'start': 1, 'length': 1}
