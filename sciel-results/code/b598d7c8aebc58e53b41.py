import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]

    # Generate primes up to 100000 to have more candidates
    primes = sieve(100000)
    
    def tau_perp(n, prime_factors):
        # prime_factors is a list of distinct primes
        k = len(prime_factors)
        if k == 0:
            return 0
        # Generate all divisors as (value, bitmask)
        divs = [(1, 0)]
        for i, p in enumerate(prime_factors):
            bit = 1 << i
            new_divs = []
            for d, m in divs:
                new_divs.append((d * p, m | bit))
            divs.extend(new_divs)
        divs.sort(key=lambda x: x[0])
        
        score = 0
        for i in range(len(divs) - 1):
            if (divs[i][1] & divs[i+1][1]) == 0:
                score += 1
        return score

    # Initialize with smallest 15 primes
    initial_primes = []
    prod = 1
    for p in primes:
        if prod * p > 10**18:
            break
        initial_primes.append(p)
        prod *= p
        
    best_n = prod
    best_score = tau_perp(prod, initial_primes)
    best_primes = list(initial_primes)

    current_primes = list(initial_primes)
    current_prod = prod
    current_score = best_score

    improved = True
    while improved and time.time() < deadline - 15.0:
        improved = False
        indices = list(range(len(current_primes)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline - 10.0:
                break
            p_old = current_primes[idx]
            
            max_p_new = int((10**18 * p_old) / current_prod)
            
            best_swap_score = -1
            best_swap_prime = -1
            best_swap_prod = -1
            
            for p_new in primes:
                if p_new > max_p_new:
                    break
                if p_new == p_old:
                    continue
                if p_new in current_primes:
                    continue
                
                new_prod = (current_prod // p_old) * p_new
                new_primes = list(current_primes)
                new_primes[idx] = p_new
                s_new = tau_perp(new_prod, new_primes)
                
                if s_new > best_swap_score:
                    best_swap_score = s_new
                    best_swap_prime = p_new
                    best_swap_prod = new_prod
            
            if best_swap_prime != -1 and best_swap_score > current_score:
                current_primes[idx] = best_swap_prime
                current_prod = best_swap_prod
                current_score = best_swap_score
                improved = True
                
                if current_score > best_score:
                    best_score = current_score
                    best_n = current_prod
                    best_primes = list(current_primes)
                    
            if time.time() > deadline - 10.0:
                break

    # Second phase: try 2-prime swaps for additional improvement
    if time.time() < deadline - 10.0:
        for _ in range(50):
            if time.time() > deadline - 10.0:
                break
            i, j = random.sample(range(len(current_primes)), 2)
            p_old_i = current_primes[i]
            p_old_j = current_primes[j]
            
            max_p_i = int((10**18 * p_old_i) / current_prod)
            max_p_j = int((10**18 * p_old_j) / current_prod)
            
            # Try swapping both
            best_pair_score = -1
            best_pair = (-1, -1)
            
            # Sample some candidates for each
            candidates_i = [p for p in primes if p <= max_p_i and p != p_old_i and p not in current_primes]
            candidates_j = [p for p in primes if p <= max_p_j and p != p_old_j and p not in current_primes]
            
            if not candidates_i or not candidates_j:
                continue
            
            # Sample up to 20 candidates each
            sample_i = random.sample(candidates_i, min(20, len(candidates_i)))
            sample_j = random.sample(candidates_j, min(20, len(candidates_j)))
            
            for pi in sample_i:
                if time.time() > deadline - 10.0:
                    break
                for pj in sample_j:
                    if pi == pj:
                        continue
                    new_primes = list(current_primes)
                    new_primes[i] = pi
                    new_primes[j] = pj
                    new_prod = current_prod // p_old_i // p_old_j * pi * pj
                    if new_prod > 10**18:
                        continue
                    s_new = tau_perp(new_prod, new_primes)
                    if s_new > best_pair_score:
                        best_pair_score = s_new
                        best_pair = (pi, pj)
            
            if best_pair[0] != -1 and best_pair_score > current_score:
                current_primes[i] = best_pair[0]
                current_primes[j] = best_pair[1]
                current_prod = current_prod // p_old_i // p_old_j * best_pair[0] * best_pair[1]
                current_score = best_pair_score
                if current_score > best_score:
                    best_score = current_score
                    best_n = current_prod
                    best_primes = list(current_primes)

    return best_n
