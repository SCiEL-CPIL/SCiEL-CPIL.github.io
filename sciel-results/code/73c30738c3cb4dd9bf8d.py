import math
import time

def _score_from_factorization(primes, exponents):
    """
    Calculate tau_perp given the prime factorization.
    primes: list of primes
    exponents: list of exponents
    """
    # Generate all divisors
    divisors = [1]
    for p, e in zip(primes, exponents):
        new_divisors = []
        power = 1
        for _ in range(e + 1):
            for d in divisors:
                new_divisors.append(d * power)
            power *= p
        divisors = new_divisors
    
    divisors.sort()
    
    count = 0
    for i in range(len(divisors) - 1):
        if math.gcd(divisors[i], divisors[i+1]) == 1:
            count += 1
            
    return count

def construct():
    """
    Returns an integer n <= 10^18 maximizing tau_perp(n).
    """
    start_time = time.time()
    TIME_LIMIT = 110  # Safety margin
    
    max_val = 10**18
    
    # First 15 primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    best_score = 0
    best_n = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 # Seed fallback
    
    # 1. Evaluate Maximal Primorial P15
    p15_val = 1
    for p in primes:
        p15_val *= p
    if p15_val <= max_val:
        score_p15 = _score_from_factorization(primes, [1]*15)
        if score_p15 > best_score:
            best_score = score_p15
            best_n = p15_val
            
    # 2. Evaluate variations of P15
    # Strategy: Remove the largest prime(s) and increase exponents of smaller primes
    # Base primes for variations: first 14 primes
    base_primes = primes[:14]
    base_val = 1
    for p in base_primes:
        base_val *= p
        
    # Try increasing exponents of 2, 3, 5, etc.
    # We have budget: max_val // base_val
    budget = max_val // base_val
    
    # Generate candidates by modifying exponents of the first few primes
    # Start with [1]*14
    # Try [2, 1, 1...], [3, 1, 1...], etc.
    # And combinations like [2, 2, 1...], [4, 1, 1...], etc.
    
    candidates = []
    
    # Helper to add a candidate
    def add_candidate(exps):
        val = 1
        for p, e in zip(base_primes, exps):
            val *= (p ** e)
        if val <= max_val:
            candidates.append((base_primes, exps, val))
            
    # 1. Single prime exponent increases
    for i in range(4): # First 4 primes
        for e in range(2, 8):
            exps = [1]*14
            exps[i] = e
            add_candidate(exps)
            
    # 2. Two prime exponent increases
    for i in range(3):
        for j in range(i+1, 4):
            for ei in range(2, 5):
                for ej in range(2, 5):
                    exps = [1]*14
                    exps[i] = ei
                    exps[j] = ej
                    add_candidate(exps)
                    
    # 3. Remove 15th prime (47) is already done in base_primes (14 primes)
    # What if we remove 14th (43) and 15th (47)?
    base_primes_13 = primes[:13]
    base_val_13 = 1
    for p in base_primes_13:
        base_val_13 *= p
    budget_13 = max_val // base_val_13
    
    # Try higher exponents on first few primes for 13-prime base
    for i in range(3):
        for e in range(2, 10):
            exps = [1]*13
            exps[i] = e
            val = 1
            for p, ex in zip(base_primes_13, exps):
                val *= (p ** ex)
            if val <= max_val:
                candidates.append((base_primes_13, exps, val))
                
    # Evaluate all candidates
    for p_list, e_list, val in candidates:
        if time.time() - start_time > TIME_LIMIT:
            break
        score = _score_from_factorization(p_list, e_list)
        if score > best_score:
            best_score = score
            best_n = val
            
    # 3. Limited DFS for other structures
    # Prioritize adding new primes (exp 1) over increasing exponents
    # This helps find P15 if it wasn't already the best, or other square-free numbers
    
    def dfs(idx, max_exp, current_n, current_primes, current_exps):
        nonlocal best_score, best_n
        
        if time.time() - start_time > TIME_LIMIT:
            return
            
        if len(current_primes) > 0:
            score = _score_from_factorization(current_primes, current_exps)
            if score > best_score:
                best_score = score
                best_n = current_n

        if idx >= len(primes):
            return
            
        p = primes[idx]
        
        # Try exponents. 
        # To find square-free numbers quickly, we should try e=1 early?
        # But standard DFS for "max divisors" tries high exponents.
        # For "max coprime pairs", we want many distinct primes.
        # So we should try e=1 (add prime) before e>1 (increase power) if we haven't added many primes yet.
        # However, the constraint is non-increasing exponents.
        # If we pick e=1, next max_exp is 1. So we can only pick e=1 or 0 for subsequent primes.
        # This perfectly generates square-free numbers!
        
        # So, if we want to find square-free numbers, we should try e=1 FIRST.
        # If we want to find high-power numbers, we try high e.
        
        # Let's try e=1 first, then other e's.
        # But we must respect max_exp.
        
        # Order: 1, then max_exp down to 2, then 0?
        # If max_exp is 1, only 1 and 0 are options.
        
        # Let's iterate e from 1 to max_exp? No, if max_exp is large, we miss the "many primes" path if we start high.
        # We want to explore the "many primes" (e=1) branch first.
        
        # But if we pick e=1, we are locked into e<=1 for all subsequent primes.
        # This finds P_k quickly.
        
        # Let's try e=1 first.
        if 1 <= max_exp:
            new_n = current_n * p
            if new_n <= max_val:
                dfs(idx + 1, 1, new_n, current_primes + [p], current_exps + [1])
                
        # Then try other exponents > 1
        for e in range(max_exp, 1, -1):
            if time.time() - start_time > TIME_LIMIT:
                return
            new_n = current_n * (p ** e)
            if new_n > max_val:
                continue
            dfs(idx + 1, e, new_n, current_primes + [p], current_exps + [e])
            
        # e=0 (skip prime)
        # If we skip, we can still use later primes?
        # In standard non-increasing exponent DFS, if we skip p_i, we can use p_j (j>i) with exp <= current max_exp?
        # No, the "non-increasing" constraint applies to the sequence of *chosen* primes.
        # If we skip p_i, the next chosen prime p_j must have exp <= (exp of previous chosen prime).
        # So we can recurse with the SAME max_exp and idx+1.
        dfs(idx + 1, max_exp, current_n, current_primes, current_exps)

    # Start DFS
    # We already evaluated P15.
    # We can start DFS with a higher max_exp to explore non-square-free numbers.
    # But we must be careful not to waste time.
    # Since we already have a strong candidate (P15), we can set a lower bar for DFS?
    # Or just let it run.
    
    dfs(0, 60, 1, [], [])
    
    return best_n

# EVOLVE-BLOCK-END
