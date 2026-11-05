"""Randomized search with trial division for phi computation."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    MAX_N = 10**12
    MAX_K = 200
    
    # Sieve primes up to sqrt(MAX_N)
    limit = 10**6
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, val in enumerate(is_prime) if val]
    
    def phi(n):
        if n == 1:
            return 1
        result = n
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                while temp % p == 0:
                    temp //= p
                result -= result // p
        if temp > 1:
            result -= result // temp
        return result

    def get_run_length(s, max_len):
        """Returns the length of the longest prefix of distinct phi values starting at s."""
        seen = set()
        for i in range(max_len):
            val = phi(s + i)
            if val in seen:
                return i
            seen.add(val)
        return max_len

    best_k = 1
    best_s = 1
    
    # Initial check for small numbers
    for s in [1, 2, 3, 4, 5]:
        k = get_run_length(s, MAX_K)
        if k > best_k:
            best_k = k
            best_s = s

    # Time-limited random search
    start_time = time.time()
    time_limit = 115  # Leave 5s for safety
    
    # Use stratified random sampling
    # Divide range into 1000 segments
    segments = 1000
    seg_size = MAX_N // segments
    
    while time.time() - start_time < time_limit:
        # Pick a random segment and a random start within it
        seg_idx = random.randint(0, segments - 1)
        base = seg_idx * seg_size
        s = base + random.randint(0, min(seg_size, MAX_N - base) - 1)
        s = max(1, s)
        
        # Ensure we don't go out of bounds
        if s + MAX_K - 1 > MAX_N:
            s = MAX_N - MAX_K + 1
        
        k = get_run_length(s, MAX_K)
        
        if k > best_k:
            best_k = k
            best_s = s
            
            # Try to extend the run
            # Check left
            if best_s > 1:
                # Check if phi(best_s - 1) is distinct from the current run
                # We need to check against all values in the run
                # Since we have the set 'seen' from get_run_length? No, it's local.
                # We can just re-check the extended run.
                new_k = get_run_length(best_s - 1, best_k + 1)
                if new_k > best_k:
                    best_k = new_k
                    best_s = best_s - 1
                    # Loop extension
                    while best_s > 1:
                        new_k = get_run_length(best_s - 1, best_k + 1)
                        if new_k > best_k:
                            best_k = new_k
                            best_s = best_s - 1
                        else:
                            break
                    
            # Check right
            if best_s + best_k - 1 < MAX_N:
                new_k = get_run_length(best_s, best_k + 1)
                if new_k > best_k:
                    best_k = new_k
                    # Loop extension
                    while best_s + best_k - 1 < MAX_N:
                        new_k = get_run_length(best_s, best_k + 1)
                        if new_k > best_k:
                            best_k = new_k
                        else:
                            break

    # Final clamp
    if best_s + best_k - 1 > MAX_N:
        best_k = int(MAX_N - best_s + 1)
    if best_k > MAX_K:
        best_k = MAX_K
    if best_k < 1:
        best_k = 1
        best_s = 1

    return {'start': int(best_s), 'length': int(best_k)}
# EVOLVE-BLOCK-END
