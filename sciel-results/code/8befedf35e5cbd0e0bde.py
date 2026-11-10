import time
import math
import random

def _tau_perp_fast(primes):
    """Compute tau_perp for a squarefree number defined by its prime list."""
    k = len(primes)
    if k == 0:
        return 0
    n_divs = 1 << k
    # Compute all divisor values
    vals = [1] * n_divs
    for i in range(k):
        p = primes[i]
        bit = 1 << i
        for m in range(n_divs):
            if m & bit:
                vals[m] = vals[m ^ bit] * p
    # Sort indices by value
    order = sorted(range(n_divs), key=lambda m: vals[m])
    # Count adjacent coprime pairs
    score = 0
    for i in range(n_divs - 1):
        a = order[i]
        b = order[i + 1]
        if (a & b) == 0:
            score += 1
    return score

def _product(primes):
    r = 1
    for p in primes:
        r *= p
    return r

def _is_prime(n):
    if n < 2:
        return False
    if n < 9:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _sieve_primes(limit):
    """Return list of primes up to limit."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, limit + 1) if sieve[i]]

def construct():
    t0 = time.time()
    DEADLINE = 150.0

    # Generate candidate primes in a reasonable range
    # For k=15, product must be <= 1e18
    # 15 primes around 100: product ~ 100^15 = 1e30, too big
    # 15 primes around 20: product ~ 20^15 = 3.2e19, still too big
    # 14 primes around 30: 30^14 ~ 4.8e20, too big
    # 13 primes around 50: 50^13 ~ 1.2e22, too big
    # Let's think: product of first 15 primes = 614889782588491410 ~ 6.1e17, OK
    # So we need primes that are not too large.
    # Let's use primes up to ~200, and try k=13,14,15
    
    all_primes = _sieve_primes(500)
    
    # We'll try different k values
    best_n = 1
    best_score = 0
    
    for k in [15, 14, 13]:
        if time.time() - t0 > DEADLINE - 10:
            break
        
        # Start with the k smallest primes that fit
        # Find the maximum prime we can use
        # Product of k primes <= 1e18
        # Try initial set: first k primes
        initial = all_primes[:k]
        prod = _product(initial)
        if prod > 10**18:
            # Need smaller primes or fewer - skip
            continue
        
        # Local search: swap one prime at a time
        current = list(initial)
        current_score = _tau_perp_fast(current)
        current_prod = prod
        
        # Candidate pool: primes from index k to some limit
        # We want to swap in larger primes to potentially improve structure
        pool_start = k
        pool_end = min(len(all_primes), pool_start + 200)
        pool = all_primes[pool_start:pool_end]
        
        # Also allow swapping with primes before index k (smaller ones)
        # But those are already in the set, so just use pool
        
        improved = True
        max_iters = 50000
        iter_count = 0
        
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            if time.time() - t0 > DEADLINE - 10:
                break
            
            # Try swapping each prime in current with each in pool
            for i in range(k):
                for j, new_p in enumerate(pool):
                    if new_p == current[i]:
                        continue
                    # Check product constraint
                    old_p = current[i]
                    new_prod = current_prod // old_p * new_p
                    if new_prod > 10**18 or new_prod < 2:
                        continue
                    # Swap
                    current[i] = new_p
                    current_prod = new_prod
                    new_score = _tau_perp_fast(current)
                    if new_score > current_score:
                        current_score = new_score
                        improved = True
                        # Update best
                        if current_score > best_score:
                            best_score = current_score
                            best_n = current_prod
                        # Restart search from here
                        break
                    else:
                        current[i] = old_p
                        current_prod = old_p * (current_prod // old_p)
                        # Revert
                        current_prod = _product(current)
                else:
                    continue
                break
        
        # After local search, update best
        if current_score > best_score:
            best_score = current_score
            best_n = current_prod
    
    return best_n
