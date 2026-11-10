import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin

    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]

    # We need primes up to a point where we can replace small primes with larger ones
    # Max prime in set could be up to 10^18 / (product of first 14 primes)
    # Product of first 14 primes is ~ 1.3e17, so max prime ~ 7.
    # Wait, product of first 15 primes is ~ 6e17.
    # If we swap p_15 (47) with a larger prime q, q can be up to 10^18 / (prod of first 14).
    # Prod first 14: 2*3*5*...*43 = 1311379765200
    # Max q = 10^18 / 1311379765200 ~ 762559.
    # So primes up to 1,000,000 should be sufficient for any single swap.
    # If we swap multiple, the bound changes, but 1,000,000 is a safe upper bound for individual primes in a 15-prime product <= 10^18.
    primes = sieve(1000000)
    
    def tau_perp(n, prime_factors):
        # prime_factors is a list of distinct primes
        divs = [1]
        for p in prime_factors:
            new_divs = []
            for d in divs:
                new_divs.append(d * p)
            divs.extend(new_divs)
        divs.sort()
        
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
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
        
    if len(initial_primes) != 15:
        # Fallback if something went wrong, though it shouldn't
        return 1

    best_n = prod
    best_score = tau_perp(prod, initial_primes)
    best_primes = list(initial_primes)

    current_primes = list(initial_primes)
    current_prod = prod
    current_score = best_score

    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        # Shuffle indices to avoid bias
        indices = list(range(len(current_primes)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline - 5.0:
                break
            p_old = current_primes[idx]
            
            # Calculate product of other primes
            prod_others = current_prod // p_old
            max_p_new = int(10**18 / prod_others)
            
            best_swap_score = -1
            best_swap_prime = -1
            best_swap_prod = -1
            
            current_set = set(current_primes)
            
            # Iterate through primes to find best swap
            # We can optimize by skipping primes that are too small or already in set
            # But checking 100k primes * 15 indices * few iterations is ~ 1.5M checks.
            # Each check involves generating divisors (32768 items) and sorting.
            # This is too slow for 1.5M full evaluations.
            
            # Optimization: We don't need to try EVERY prime.
            # We can sample or restrict the search range.
            # Or, we can just try a subset of candidates.
            # Let's try a random subset of valid candidates, or just the next few valid ones?
            # Actually, the landscape might be smooth.
            # Let's try to limit the number of candidates per swap to e.g. 50 random valid primes.
            
            valid_candidates = []
            # Find valid candidates range
            # p_new must be > 1, not in current_set, and <= max_p_new
            # To make it fast, we can just iterate and collect a few
            # Or better, just iterate and keep the best, but break early if time is tight?
            # No, we need to find the BEST swap to ensure improvement.
            
            # Alternative: Pre-compute divisors for the base set?
            # If we replace p_old with p_new, the set of divisors changes completely.
            
            # Let's reduce the search space.
            # We only need to consider primes that are significantly different?
            # Let's just try a limited number of candidates, say 100, sampled from the valid range.
            
            # First, find the index of p_old in primes list to know where to start/stop?
            # Not strictly necessary if we just scan.
            
            # Let's scan but limit to 200 candidates to keep speed up.
            # If we don't find an improvement in 200 random-ish candidates, we might miss it.
            # But we do multiple passes.
            
            # Better: Scan all valid candidates but make tau_perp faster?
            # tau_perp is the bottleneck.
            
            # Let's try a hybrid: Scan all valid candidates, but only calculate tau_perp if the product is close to the limit or if the prime is "interesting"?
            # No, just limit the candidates.
            
            # Let's collect valid primes into a list first? No, that's memory.
            # Just iterate.
            
            count = 0
            # We will check up to 500 candidates per swap position
            for p_new in primes:
                if p_new > max_p_new:
                    break
                if p_new in current_set:
                    continue
                if p_new == p_old:
                    continue
                
                count += 1
                if count > 500:
                    break
                
                new_prod = prod_others * p_new
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

    return best_n
