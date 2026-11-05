import math
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # Primes up to 60
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]

    def get_divisors(n, factors):
        """
        factors: list of (prime, exponent)
        """
        divs = [1]
        for p, e in factors:
            new_divs = []
            power = 1
            for _ in range(e):
                power *= p
                for d in divs:
                    new_divs.append(d * power)
            divs += new_divs
        return sorted(divs)

    def calc_score(n, factors):
        if n < 2:
            return 0
        divs = get_divisors(n, factors)
        if len(divs) < 2:
            return 0
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i + 1]) == 1:
                score += 1
        return score

    def make_factors(exponents, num_primes):
        """
        exponents: list of exponents for the first num_primes primes
        """
        factors = []
        for i in range(num_primes):
            p = primes[i]
            e = exponents[i]
            if e > 0:
                factors.append((p, e))
        return factors

    def product_of_factors(factors):
        n = 1
        for p, e in factors:
            n *= (p ** e)
        return n

    candidates = []

    # 1. Square-free with 15 primes
    # Primes 2..47
    exps_15_sqf = [1] * 15
    factors_15 = make_factors(exps_15_sqf, 15)
    n_15 = product_of_factors(factors_15)
    if n_15 <= 10**18:
        candidates.append((n_15, factors_15))

    # 2. Square-free with 14 primes (allowing some exponents > 1? No, start with sqf)
    # Actually, let's just do local search from the 15-prime square-free number.
    # And also from a highly composite number.

    # Find a highly composite number near 10^18
    # Heuristic: exponents decreasing
    # Try 2^10 3^5 5^3 7^2 11^1 ...
    # This is the standard max tau structure.
    
    # Let's generate a few "max tau" style candidates
    # We'll do a small search for exponents e1>=e2>=...
    
    def search_exponents(num_primes, max_exp_start=15):
        """
        Recursive search for exponents e1 >= e2 >= ... >= ek >= 1
        such that product <= 10^18.
        """
        best_local = None
        def backtrack(idx, current_product, exponents):
            nonlocal best_local
            if time.time() > deadline:
                return
            if idx == num_primes:
                # Check if this is a good candidate
                # We just collect it, evaluation happens later
                # To limit candidates, we only keep if product is "close" to 10^18?
                # Or just keep the best tau?
                # For now, let's just return the exponents
                # We'll evaluate all valid ones found
                candidates.append((current_product, make_factors(exponents, num_primes)))
                return
            
            p = primes[idx]
            # Max exponent allowed by product
            max_e = max_exp_start
            if idx == 0:
                # Estimate max e for p
                # p^e <= 10^18 / (product of min others)
                # Simplified: just try down from max_exp_start
                pass
            
            prev_e = exponents[idx-1] if idx > 0 else max_exp_start
            # Try exponents from min(prev_e, max_exp_start) down to 1
            for e in range(min(prev_e, max_exp_start), 0, -1):
                new_prod = current_product * (p ** e)
                if new_prod > 10**18:
                    continue
                # Pruning: if new_prod * (product of smallest possible remaining) > 10^18, skip?
                # Smallest remaining product is 1 (if we allow e=0? No, we require e>=1 for distinct primes in this branch)
                # Actually, for max tau, we usually want all primes included.
                # So min product of remaining is product of primes[idx+1:]
                # If new_prod * prod(remaining) > 10^18, then even with e=1 for all remaining, it's too big.
                # So we can prune.
                
                # Calculate min product of remaining primes (each with exp 1)
                min_rem_prod = 1
                for j in range(idx + 1, num_primes):
                    min_rem_prod *= primes[j]
                
                if new_prod * min_rem_prod > 10**18:
                    # Even with 1 for all remaining, it's too big.
                    # Since we are decreasing e, new_prod will only get smaller.
                    # So we should continue to smaller e.
                    # Wait, if e is smaller, new_prod is smaller, so it might fit.
                    # So we don't break, we continue.
                    pass
                
                exponents.append(e)
                backtrack(idx + 1, new_prod, exponents)
                exponents.pop()
        
        backtrack(0, 1, [])

    # Run search for 15 primes (square-free is a subset of this, but this allows higher exponents)
    # This search might be too slow if not pruned well.
    # Let's limit the search depth or use a smarter heuristic.
    
    # Alternative: Just generate a few specific candidates.
    
    # Candidate 2: 2^10 3^5 5^3 7^2 11 13 17 19 23 29 31 37 41 43
    # 14 primes.
    exps_14_hc = [10, 5, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    factors_14 = make_factors(exps_14_hc, 14)
    n_14 = product_of_factors(factors_14)
    if n_14 <= 10**18:
        candidates.append((n_14, factors_14))

    # Candidate 3: 2^9 3^4 5^2 7^2 11 13 17 19 23 29 31 37 41 43 47
    # 15 primes.
    exps_15_hc = [9, 4, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    factors_15_hc = make_factors(exps_15_hc, 15)
    n_15_hc = product_of_factors(factors_15_hc)
    if n_15_hc <= 10**18:
        candidates.append((n_15_hc, factors_15_hc))

    # Candidate 4: 2^8 3^4 5^3 7^2 11 13 17 19 23 29 31 37 41 43
    exps_14_hc2 = [8, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    factors_14_hc2 = make_factors(exps_14_hc2, 14)
    n_14_hc2 = product_of_factors(factors_14_hc2)
    if n_14_hc2 <= 10**18:
        candidates.append((n_14_hc2, factors_14_hc2))

    # Evaluate initial candidates
    best_n = 2
    best_score = -1
    best_factors = [(2, 1)]
    
    for n, factors in candidates:
        if time.time() > deadline:
            break
        s = calc_score(n, factors)
        if s > best_score:
            best_score = s
            best_n = n
            best_factors = factors

    # Local Search on best_factors
    # Move: Change exponent of a prime by +/- 1
    # Constraint: product <= 10^18
    # Also try swapping primes?
    # For now, just exponent search.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_n = best_n
        current_factors = best_factors
        current_score = best_score
        
        # Try changing each exponent
        for i in range(len(current_factors)):
            if time.time() > deadline:
                break
            p, e = current_factors[i]
            # Try e+1
            new_e = e + 1
            new_prod = current_n // (p ** e) * (p ** new_e)
            if new_prod <= 10**18:
                new_factors = current_factors.copy()
                new_factors[i] = (p, new_e)
                s = calc_score(new_prod, new_factors)
                if s > current_score:
                    best_n = new_prod
                    best_factors = new_factors
                    best_score = s
                    improved = True
                    # Restart loop to update base
                    break
            
            # Try e-1 (if e > 1)
            if e > 1:
                new_e = e - 1
                new_prod = current_n // (p ** e) * (p ** new_e)
                # new_prod is always <= 10^18
                new_factors = current_factors.copy()
                new_factors[i] = (p, new_e)
                s = calc_score(new_prod, new_factors)
                if s > current_score:
                    best_n = new_prod
                    best_factors = new_factors
                    best_score = s
                    improved = True
                    break

    return best_n
