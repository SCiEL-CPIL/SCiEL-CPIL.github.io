import math
import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s limit

    def time_left():
        return DEADLINE - (time.time() - start_time)

    def sieve_primes(limit):
        """Sieve of Eratosthenes to find all primes up to limit."""
        if limit < 2:
            return []
        is_prime = bytearray([1]) * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        return [i for i in range(2, limit + 1) if is_prime[i]]

    # Precompute primes up to sqrt(10^12) = 10^6
    # This takes < 0.1s
    primes = sieve_primes(10**6)

    best_k = 1
    best_s = 1

    # We will search in blocks.
    # Block size 100,000 is a good trade-off.
    # Smaller blocks are faster to process but might miss long runs if they span blocks.
    # However, runs of length > 200 are unlikely to span a 100,000 block boundary randomly.
    # Also, we can just check within the block.
    # If a run spans a boundary, we might miss it, but for large N, collisions are rare,
    # so runs are likely short or isolated.
    # Actually, we want to MAXIMIZE k. The theoretical max is likely small (e.g., < 20).
    # So a block size of 10,000 is probably sufficient and faster.
    # Let's use 10,000 to allow more blocks to be scanned in the time limit.
    
    block_size = 10000
    
    # Start searching from a large base to minimize collisions
    # 10^11 is a good start.
    base = 10**11
    
    # We will scan blocks forward from base
    # We stop when we hit 10^12 or time runs out.
    
    lo = base
    while lo + block_size <= 10**12:
        if time_left() < 5.0:
            break
            
        hi = lo + block_size - 1
        
        # Compute phi for [lo, hi]
        n = hi - lo + 1
        phi = list(range(lo, hi + 1))
        remaining = list(range(lo, hi + 1))
        
        # Handle phi(1) if lo == 1 (not the case here since base is large)
        if lo == 1:
            phi[0] = 1
            remaining[0] = 1

        # Segmented sieve
        # Only primes p where p*p <= hi are needed for the inner loop?
        # No, we need all primes p <= hi to divide remaining.
        # But for the "update phi" step, we only need to check if p divides the original number.
        # The standard algorithm:
        # For each prime p:
        #   For each multiple m of p in [lo, hi]:
        #     If remaining[m] % p == 0:
        #       phi[m] -= phi[m] // p
        #       while remaining[m] % p == 0:
        #         remaining[m] //= p
        # After all primes:
        #   For each i:
        #     If remaining[i] > 1:
        #       phi[i] -= phi[i] // remaining[i]
        
        # Optimization: We only need to iterate primes p <= sqrt(hi) for the inner loop?
        # No. If p > sqrt(hi), it can only divide the number once.
        # If we skip primes p > sqrt(hi) in the main loop, they will be caught in the "remaining > 1" step.
        # So we can limit the main loop to p <= sqrt(hi).
        # This reduces the number of primes we iterate over significantly.
        
        limit_p = int(math.isqrt(hi))
        
        for p in primes:
            if p > limit_p:
                break
            
            # Find first multiple of p in [lo, hi]
            start = ((lo + p - 1) // p) * p
            if start > hi:
                continue
                
            for j in range(start, hi + 1, p):
                idx = j - lo
                if remaining[idx] % p == 0:
                    phi[idx] -= phi[idx] // p
                    while remaining[idx] % p == 0:
                        remaining[idx] //= p

        # Handle remaining prime factors
        for i in range(n):
            if remaining[i] > 1:
                phi[i] -= phi[i] // remaining[i]

        # Find longest run of distinct values in phi
        # Sliding window
        seen = {}
        left = 0
        max_k_block = 1
        max_start_block = lo
        
        for right in range(n):
            val = phi[right]
            seen[val] = seen.get(val, 0) + 1
            
            while seen[val] > 1:
                left_val = phi[left]
                seen[left_val] -= 1
                left += 1
                
            current_k = right - left + 1
            if current_k > max_k_block:
                max_k_block = current_k
                max_start_block = lo + left
                
            # If we found a run of 200, we can break early?
            # The task says 1 <= k <= 200.
            # If we find 200, we can't do better in this block.
            if max_k_block >= 200:
                break

        if max_k_block > best_k:
            best_k = max_k_block
            best_s = max_start_block
            
        # Move to next block
        lo = hi + 1

    # Ensure validity
    if best_k < 1:
        best_k = 1
        best_s = 1
    if best_s + best_k - 1 > 10**12:
        best_k = min(best_k, 10**12 - best_s + 1)
    if best_k < 1:
        best_k = 1
    if best_s < 1:
        best_s = 1

    return {'start': best_s, 'length': best_k}
