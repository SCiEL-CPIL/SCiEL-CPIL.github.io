import time
import math

def _tau_perp_squarefree(primes):
    """
    Calculate tau_perp for a square-free number given its list of prime factors.
    For square-free n with k primes, divisors are all 2^k subsets.
    We generate divisors, sort them, and count adjacent coprime pairs.
    """
    k = len(primes)
    # Generate divisors using bitmasks
    divs = []
    for mask in range(1 << k):
        val = 1
        for i in range(k):
            if mask & (1 << i):
                val *= primes[i]
        divs.append(val)
    
    divs.sort()
    
    score = 0
    prev = divs[0]
    for curr in divs[1:]:
        if math.gcd(prev, curr) == 1:
            score += 1
        prev = curr
    return score

def _factor_squarefree(n):
    """Factor a square-free integer into its prime factors."""
    factors = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return sorted(factors)

def construct():
    start_time = time.time()
    LIMIT = 10**18
    DEADLINE = start_time + 145.0  # Safety margin before 150s kill
    
    # List of first ~40 primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 
              53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 
              109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173]
    
    # The maximum number of distinct prime factors for n <= 10^18 is 15.
    # Product of first 15 primes: 2*3*...*47 = 614889782588491410 ~ 6.15 * 10^17
    # Product of first 16 primes: * 53 = 3.26 * 10^19 > 10^18
    # So we are looking for sets of 15 distinct primes.
    
    # Strategy:
    # 1. Start with the standard set: first 15 primes.
    # 2. Use local search: try swapping each prime in the set with a larger prime
    #    (not already in the set) such that the product remains <= LIMIT.
    # 3. Iterate until no improvement or deadline reached.
    
    # Base set: first 15 primes
    std_set = primes[:15]
    n_std = 1
    for p in std_set:
        n_std *= p
    if n_std > LIMIT:
        # Fallback: use fewer primes if needed (shouldn't happen)
        while n_std > LIMIT and len(std_set) > 1:
            std_set = std_set[:-1]
            n_std = 1
            for p in std_set:
                n_std *= p
    
    best_n = n_std
    best_score = _tau_perp_squarefree(std_set)
    current_set = std_set[:]
    
    # Local search: try single swaps
    improved = True
    while improved and time.time() < DEADLINE:
        improved = False
        current_score = _tau_perp_squarefree(current_set)
        
        # Try swapping each prime in current_set with a larger prime
        for idx, p_old in enumerate(current_set):
            if time.time() > DEADLINE - 5:
                break
            
            # Find candidate replacement primes > p_old
            # We need to check if p_new is in current_set. If so, skip.
            # Also product / p_old * p_new <= LIMIT.
            
            for p_new in primes:
                if p_new <= p_old:
                    continue
                if p_new in current_set:
                    continue
                
                new_product = best_n // p_old * p_new
                if new_product > LIMIT:
                    break # Since primes are sorted, larger p_new will also exceed limit
                
                new_set = current_set.copy()
                new_set[idx] = p_new
                new_set.sort()
                
                score_new = _tau_perp_squarefree(new_set)
                if score_new > best_score:
                    best_score = score_new
                    best_n = new_product
                    current_set = new_set
                    improved = True
                    # Restart the while loop with the new best set
                    break
            
            if improved:
                break
        
        # If no improvement in a full pass, we might be at a local maximum.
        # Try a few random restarts if we have time?
        # Or try swapping two primes simultaneously?
        # For now, stick with single swaps.
        
        if time.time() > DEADLINE - 10:
            break

    # Final check
    if best_n == 0 or best_n < 2:
        # Fallback to first 15 primes
        best_n = 1
        for p in primes[:15]:
            best_n *= p
    
    assert 2 <= best_n <= LIMIT
    return best_n
