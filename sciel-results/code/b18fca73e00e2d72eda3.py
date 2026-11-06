<<<<<<< SEARCH
    # Phase 5: Try 14 primes with one prime cubed (p^3)
    if time.time() > deadline:
        return best_n
    
    for cube_idx in range(4):  # Cube primes at index 0-3
        if time.time() > deadline:
            break
        cube_prime = base_primes[cube_idx]
        cube_prod = cube_prime ** 3
        
        other_primes = []
        prod = cube_prod
        for p in base_primes[:20]:
            if p == cube_prime:
                continue
            if prod * p > 10**18:
                break
            other_primes.append(p)
            prod *= p
            if len(other_primes) == 13:
                break
        
        if len(other_primes) == 13:
            primes_list = [cube_prime] + other_primes
            exponents = [3] + [1]*13
            n = prod
            score = _tau_perp_bitset(primes_list, exponents)
            if score > best_score:
                best_score = score
                best_n = n
    
    return best_n
=======
    # Phase 5: Try 14 primes with one prime cubed (p^3)
    if time.time() > deadline:
        return best_n
    
    for cube_idx in range(4):  # Cube primes at index 0-3
        if time.time() > deadline:
            break
        cube_prime = base_primes[cube_idx]
        cube_prod = cube_prime ** 3
        
        other_primes = []
        prod = cube_prod
        for p in base_primes[:20]:
            if p == cube_prime:
                continue
            if prod * p > 10**18:
                break
            other_primes.append(p)
            prod *= p
            if len(other_primes) == 13:
                break
        
        if len(other_primes) == 13:
            primes_list = [cube_prime] + other_primes
            exponents = [3] + [1]*13
            n = prod
            score = _tau_perp_bitset(primes_list, exponents)
            if score > best_score:
                best_score = score
                best_n = n
                
    # Phase 6: Check for "separated" 13-prime case
    # If we use 12 small primes and 1 large prime p > P_12, the divisors separate.
    # Score = 2 * Score(12) + 1
    if time.time() > deadline:
        return best_n
        
    p12 = base_primes[:12]
    n12 = 1
    for p in p12:
        n12 *= p
        
    score12 = _tau_perp_bitset(p12, [1]*12)
    target_score = 2 * score12 + 1
    
    if target_score > best_score:
        # Find a prime p such that n12 * p <= 10^18 and p > n12
        max_p = 10**18 // n12
        # We need p > n12
        # n12 is approx 7.7e9
        # max_p is approx 1.3e8
        # So we need a prime in (n12, max_p)
        # Since n12 is large, we can just pick the smallest prime > n12
        # We don't have primes that high in all_primes (max 10000)
        # We need to find a prime in that range.
        # Simple trial division or Miller-Rabin
        start = n12 + 1
        # Ensure start is odd
        if start % 2 == 0:
            start += 1
        found_p = None
        for candidate in range(start, min(start + 10000, max_p + 1)):
            if _is_prime(candidate):
                found_p = candidate
                break
        if found_p:
            n_candidate = n12 * found_p
            # Verify score
            # We can use the formula 2*score12+1 if we are sure about separation
            # But to be safe, compute it?
            # Computing it requires generating divisors of n_candidate.
            # n_candidate has 13 prime factors. 2^13 = 8192 divisors. Fast.
            primes_list = p12 + [found_p]
            exponents = [1]*13
            score = _tau_perp_bitset(primes_list, exponents)
            if score > best_score:
                best_score = score
                best_n = n_candidate

    return best_n
>>>>>>> REPLACE
