"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    start_time = time.time()
    TIME_LIMIT = 145.0  # safety margin before 150s kill

    def time_left():
        return TIME_LIMIT - (time.time() - start_time)

    # Precompute primes up to 10^6 using sieve
    def sieve_primes(lim):
        is_prime = bytearray(b'\x01') * (lim + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(math.isqrt(lim)) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
        return [i for i in range(2, lim + 1) if is_prime[i]]

    primes = sieve_primes(10**6)

    def phi_fast(n):
        result = n
        m = n
        for p in primes:
            if p * p > m:
                break
            if m % p == 0:
                result -= result // p
                while m % p == 0:
                    m //= p
        if m > 1:
            result -= result // m
        return result

    def find_run(s, max_k=200):
        seen = set()
        k = 0
        for i in range(max_k):
            v = phi_fast(s + i)
            if v in seen:
                break
            seen.add(v)
            k += 1
        return k

    best = (1, 1)

    # Phase 1: Segmented sieve approach for a large contiguous block
    # Compute phi for a block of size ~5 million using segmented factorization
    BLOCK_SIZE = 5 * 10**6
    BLOCK_START = 1
    
    # For each number in the block, we need to factor it.
    # We use the primes list to divide out small factors.
    # This is essentially a segmented sieve for phi.
    
    # Initialize arrays for the block
    # rem[i] will hold the remaining part of (BLOCK_START + i) after dividing by small primes
    # phi_val[i] will hold the current phi value
    rem = list(range(BLOCK_START, BLOCK_START + BLOCK_SIZE))
    phi_val = list(range(BLOCK_START, BLOCK_START + BLOCK_SIZE))
    
    # For each prime p, mark multiples in the block
    for p in primes:
        if p * p > BLOCK_START + BLOCK_SIZE:
            break
        # Find the first multiple of p in the block
        first_mult = ((BLOCK_START + p - 1) // p) * p
        if first_mult < BLOCK_START:
            first_mult = BLOCK_START
        for m in range(first_mult, BLOCK_START + BLOCK_SIZE, p):
            idx = m - BLOCK_START
            if rem[idx] % p == 0:
                phi_val[idx] -= phi_val[idx] // p
                while rem[idx] % p == 0:
                    rem[idx] //= p
    
    # Now handle remaining factors (which must be prime or 1)
    phi_block = [0] * BLOCK_SIZE
    for i in range(BLOCK_SIZE):
        r = rem[i]
        if r > 1:
            phi_block[i] = phi_val[i] - phi_val[i] // r
        else:
            phi_block[i] = phi_val[i]
    
    # Scan for the longest run of distinct phi values in the block
    seen = set()
    k = 0
    for i in range(BLOCK_SIZE):
        v = phi_block[i]
        if v in seen:
            # Reset the run
            if k > best[0]:
                best = (k, BLOCK_START + i - k + 1)
            seen = set()
            k = 0
        seen.add(v)
        k += 1
        # Cap k at 200 for efficiency, but we need to track the actual run length
        # Actually, we should just keep going but only care about runs up to 200
        # If k > 200, we can stop tracking this specific run's extension beyond 200
        # but we still need to clear 'seen' when a duplicate is found.
        # However, if k > 200, the maximum possible answer is 200, so we can just record 200 and move on?
        # No, because a duplicate might break the run later.
        # But since max_k is 200, if we have a run of length > 200, the answer is at least 200.
        # We can just record 200 and reset? No, we need to find the exact start.
        # Let's just track it properly.
        if k > best[0]:
            best = (k, BLOCK_START + i - k + 1)
    
    if k > best[0]:
        best = (k, BLOCK_START + BLOCK_SIZE - k)

    # Phase 2: Random sampling in [1, 10^12]
    # Focus on areas where runs might be longer.
    # Heuristic: runs of distinct phi values are more likely when numbers are "typical".
    # Random sampling is a reasonable fallback.
    random.seed(42)
    attempts = 0
    while time_left() > 5:
        # Sample from a wider range, but avoid the block we already scanned
        s = random.randint(1, 10**12 - 200)
        # Skip if we already scanned this region in Phase 1
        if BLOCK_START <= s <= BLOCK_START + BLOCK_SIZE - 200:
            continue
        k = find_run(s)
        if k > best[0]:
            best = (k, s)
        attempts += 1
        # Occasionally, try a local search around the best found so far
        if attempts % 100 == 0 and best[0] > 1:
            # Try a few neighbors of the current best
            for delta in range(-10, 11):
                s_new = best[1] + delta
                if s_new >= 1 and s_new + 200 <= 10**12:
                    k_new = find_run(s_new)
                    if k_new > best[0]:
                        best = (k_new, s_new)

    return {'start': best[1], 'length': best[0]}
# EVOLVE-BLOCK-END
