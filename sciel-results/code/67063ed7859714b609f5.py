import time
import math
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # safety margin before 120s limit

    # Precompute primes up to a reasonable limit using sieve
    SIEVE_LIMIT = 200000
    is_prime_sieve = bytearray(b'\x01') * (SIEVE_LIMIT + 1)
    is_prime_sieve[0] = is_prime_sieve[1] = 0
    for i in range(2, int(math.isqrt(SIEVE_LIMIT)) + 1):
        if is_prime_sieve[i]:
            for j in range(i*i, SIEVE_LIMIT + 1, i):
                is_prime_sieve[j] = 0
    primes = np.array([i for i in range(2, SIEVE_LIMIT + 1) if is_prime_sieve[i]], dtype=np.int64)
    
    # Vectorized p-smooth check for a batch of numbers p^2+1 .. p^2+K
    # Returns the length of the longest prefix that is p-smooth
    def find_max_k_vectorized(p, max_k=200):
        p2 = p * p
        # Create array of numbers to check: p^2+1, p^2+2, ..., p^2+max_k
        # We'll track which are still "alive" (not yet proven to have a factor > p)
        # Start with all values
        vals = p2 + np.arange(1, max_k + 1, dtype=np.int64)
        alive = np.ones(max_k, dtype=bool)
        
        # For each prime q <= p, divide out all occurrences
        # Only process primes up to min(p, sqrt(max_val))
        max_val = p2 + max_k
        sqrt_max = int(math.isqrt(max_val))
        q_limit = min(p, sqrt_max)
        
        # Get primes up to q_limit
        mask = primes <= q_limit
        small_primes = primes[mask]
        
        for q in small_primes:
            if not np.any(alive):
                break
            # Divide out q from all alive values
            for idx in np.where(alive)[0]:
                while vals[idx] % q == 0 and vals[idx] > 1:
                    vals[idx] //= q
            # After division, check which alive values now have a remaining factor > p
            # A value is "bad" if its remaining value > p and > 1
            # Actually: after dividing by all primes <= q, if remaining > 1:
            #   - if remaining <= p: it's prime and <= p, so OK
            #   - if remaining > p: it has a prime factor > p, so BAD
            # But we can't determine this until we've tried all primes up to sqrt(remaining)
            # So we continue the loop; at the end, check remaining values
            
        # After processing all primes up to q_limit, check remaining values
        # For each alive value, if it's > 1:
        #   - if it's <= p: it's prime (or 1) and <= p, so smooth
        #   - if it's > p: it's a prime > p (since we divided by all primes <= sqrt(max_val)), so NOT smooth
        # Wait, we only divided by primes up to q_limit = min(p, sqrt_max)
        # If p < sqrt_max, we stopped at p. Remaining could be composite with factors > p.
        # If p >= sqrt_max, we stopped at sqrt_max, so remaining is 1 or prime.
        
        # Better approach: after dividing by all primes <= p, if remaining > 1:
        #   remaining is prime (since we divided by all primes up to min(p, sqrt(max_val)))
        #   if p >= sqrt(max_val): remaining is 1 or prime, and if prime it's <= max_val
        #   Actually this is getting complex. Let me use a simpler but still vectorized approach.
        
        # Simpler: just do scalar checks but with early termination and numpy for the initial batch
        k = 0
        for j in range(1, max_k + 1):
            n = p2 + j
            if n <= p:
                k += 1
                continue
            rem = n
            smooth = True
            for q in small_primes:
                if q > p:
                    break
                if q * q > rem:
                    break
                while rem % q == 0:
                    rem //= q
            if rem > p:
                smooth = False
            if not smooth:
                break
            k += 1
        return k

    best_k = 1
    best_p = 2

    # Search strategy: test primes, but skip very small ones quickly
    # and focus computational effort where it's most likely to help
    num_primes = len(primes)
    
    # Phase 1: Quick scan of all primes up to SIEVE_LIMIT
    for idx in range(num_primes):
        if time.time() - start_time > DEADLINE:
            break
        p = int(primes[idx])
        k = find_max_k_vectorized(p)
        if k > best_k:
            best_k = k
            best_p = p

    return {'p': best_p, 'k': best_k}
