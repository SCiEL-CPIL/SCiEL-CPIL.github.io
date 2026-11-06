import math
import time

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s limit
    
    LIMIT = 10**18
    
    # First 16 primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    
    def get_divisors_from_factors(factors):
        """Generate all divisors from list of (prime, exponent) pairs."""
        divs = [1]
        for p, e in factors:
            new_divs = []
            for d in divs:
                val = d
                for _ in range(e + 1):
                    new_divs.append(val)
                    val *= p
            divs = new_divs
        return sorted(divs)
    
    def tau_perp_from_factors(factors):
        """Count adjacent coprime pairs in sorted divisors."""
        divs = get_divisors_from_factors(factors)
        count = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                count += 1
        return count
    
    best_n = 2
    best_score = 0
    
    def try_candidate(factors):
        nonlocal best_n, best_score
        n = 1
        for p, e in factors:
            n *= p ** e
        if n > LIMIT or n < 2:
            return
        # Optimization: If we have seen a similar structure or if time is tight, 
        # we still compute score. The number of candidates is small enough.
        score = tau_perp_from_factors(factors)
        if score > best_score:
            best_score = score
            best_n = n

    # Strategy: 
    # We want to maximize the number of distinct prime factors to create a "dense" 
    # set of divisors where coprime adjacencies are likely.
    # We then use the remaining budget to increase exponents of the smallest primes.
    
    # Precompute products of first k primes
    # P_k = product(primes[:k])
    
    # Iterate over number of distinct primes k
    # Max k is such that product(primes[:k]) <= 10^18
    # product of first 15 primes: 614889782588491410 ~ 6.1e17
    # product of first 16 primes: 32589158477190044730 ~ 3.2e19 (Too big)
    # So max distinct primes is 15.
    
    max_k = 15
    
    # We will search for the best distribution of exponents for the first few primes
    # given a base set of k distinct primes.
    
    for k in range(max_k, 9, -1):
        if time.time() - start_time > DEADLINE:
            break
            
        base_primes = primes[:k]
        base_product = 1
        for p in base_primes:
            base_product *= p
            
        if base_product > LIMIT:
            continue
            
        remaining_budget = LIMIT // base_product
        
        # We will assign extra exponents to the first few primes in base_primes.
        # Let's say we allow extra exponents for primes 0, 1, 2, 3 (i.e., 2, 3, 5, 7).
        # The rest will have exponent 1.
        
        # We need to find e_0, e_1, e_2, e_3 such that:
        # 2^(1+e_0) * 3^(1+e_1) * 5^(1+e_2) * 7^(1+e_3) * (product of rest) <= LIMIT
        # This is equivalent to:
        # 2^e_0 * 3^e_1 * 5^e_2 * 7^e_3 <= remaining_budget
        
        # We iterate over possible extra exponents.
        # Max extra exponent for 2: log2(remaining_budget)
        # We limit the depth of recursion/loops to keep it fast.
        
        # Helper to generate candidates for a fixed set of "flexible" primes
        # flexible_indices = [0, 1, 2, 3]
        
        def search_exponents(idx, current_mult, current_factors_list):
            if time.time() - start_time > DEADLINE:
                return
            
            if idx == 4: # We only flex first 4 primes
                # Construct final factors
                # current_factors_list has extra exponents for first 4
                # The rest (from index 4 to k-1) have extra exponent 0 (i.e., total exp 1)
                final_factors = []
                for i in range(k):
                    if i < 4:
                        # current_factors_list[i] is the EXTRA exponent
                        # Total exponent is 1 + extra
                        final_factors.append((primes[i], 1 + current_factors_list[i]))
                    else:
                        final_factors.append((primes[i], 1))
                try_candidate(final_factors)
                return
            
            p = primes[idx]
            # Max extra exponent for this prime given current_mult
            # current_mult * p^extra <= remaining_budget
            max_extra = 0
            temp = remaining_budget // current_mult
            while temp >= p:
                temp //= p
                max_extra += 1
                
            # Try extra exponents from max_extra down to 0
            # We prioritize higher exponents on smaller primes? 
            # Or just iterate all.
            # To save time, we can limit max_extra to a reasonable number, 
            # e.g., 20 for 2, 10 for 3, etc. But the while loop calculates it.
            
            # Heuristic: Iterate downwards to find good candidates early?
            # Or upwards? 
            # Usually, distributing "mass" to smaller primes is good for divisor count,
            # but for tau_perp, we want many distinct primes. 
            # Since we fixed k, we are just optimizing the shape.
            
            for extra in range(max_extra + 1):
                new_mult = current_mult * (p ** extra)
                if new_mult > remaining_budget:
                    break
                search_exponents(idx + 1, new_mult, current_factors_list + [extra])
            
        # Start search with idx=0, current_mult=1, empty extra list
        search_exponents(0, 1, [])
        
    # Fallback: if no candidate was found (should not happen), return 2
    if best_n == 2 and best_score == 0:
        # Try a simple squarefree number
        try_candidate([(p, 1) for p in primes[:15]])
        
    return best_n

# EVOLVE-BLOCK-END
