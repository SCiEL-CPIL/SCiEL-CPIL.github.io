import math
import time

def construct():
    start_time = time.time()
    time_limit = 140 # Safety margin for 150s limit
    
    max_val = 10**18
    # Primes up to 59. Product of first 16 primes is ~3.2e18 > 1e18.
    # So max 15 distinct primes.
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    
    candidates = set() # Use set to avoid duplicates if we add multiple ways
    
    # 1. DFS for Highly Composite Numbers (Non-increasing exponents)
    def dfs(idx, max_exp, current_n, current_tau):
        if time.time() - start_time > 60: # Stop generating after 60s
            return
            
        if current_n > 1:
            candidates.add((current_tau, current_n))
            
        if idx >= len(primes):
            return
            
        p = primes[idx]
        limit_n = max_val // current_n
        
        # Calculate max exponent for this prime
        e_max = 0
        temp = 1
        while temp * p <= limit_n and e_max < max_exp:
            temp *= p
            e_max += 1
            
        for e in range(1, e_max + 1):
            next_n = current_n * (p ** e)
            next_tau = current_tau * (e + 1)
            dfs(idx + 1, e, next_n, next_tau)
            
            if time.time() - start_time > 60:
                return

    dfs(0, 60, 1, 1)
    
    # 2. Explicitly add Primorials and small variations
    # P_k = product of first k primes
    pk = 1
    for i, p in enumerate(primes):
        pk *= p
        if pk > max_val:
            break
        candidates.add((2**(i+1), pk)) # tau is 2^(i+1)
        
        # Add P_k * 2^a for small a
        for a in range(1, 5):
            n = pk * (2 ** a)
            if n <= max_val:
                candidates.add((2**(i+1) * (a+1), n))
                
        # Add P_k * 3^a
        for a in range(1, 4):
            n = pk * (3 ** a)
            if n <= max_val:
                candidates.add((2**(i+1) * (a+1), n))

    if not candidates:
        return 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43 * 47

    # Convert to list
    candidate_list = list(candidates)
    
    # Sort by tau descending
    candidate_list.sort(key=lambda x: x[0], reverse=True)
    
    def tau_perp(n):
        # Factorize n using small primes
        factors = []
        temp_n = n
        for p in primes:
            if temp_n % p == 0:
                count = 0
                while temp_n % p == 0:
                    temp_n //= p
                    count += 1
                factors.append((p, count))
            if temp_n == 1:
                break
        
        if temp_n > 1:
            factors.append((temp_n, 1))

        # Generate divisors
        divs = [1]
        for p, e in factors:
            current_divs = list(divs)
            factor = 1
            for _ in range(e):
                factor *= p
                for d in current_divs:
                    divs.append(d * factor)
                    
        divs.sort()
        
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score

    best_score = -1
    best_n = candidate_list[0][1]
    
    # Evaluate top candidates
    # We want to evaluate as many as possible.
    # Limit to top 5000 by tau, but also ensure primorials are evaluated.
    # Since we added primorials explicitly, they are in the list.
    # But they might have lower tau than HCNs.
    # So we evaluate the top 5000 by tau.
    # And we also evaluate ALL primorials (which are likely in the top 5000 anyway if k is large, 
    # but for small k, tau is small. E.g. P_5 has tau 32. HCNs have tau 1000s.
    # So P_5 might not be in top 5000.
    # But P_5 is unlikely to be the global max. The global max is likely large k.
    # So top 5000 by tau is a good heuristic.
    
    eval_limit = 5000
    evaluated = set()
    
    for i, (tau, n) in enumerate(candidate_list):
        if i >= eval_limit:
            break
        if time.time() - start_time > time_limit - 5:
            break
        if n in evaluated:
            continue
        evaluated.add(n)
        
        score = tau_perp(n)
        if score > best_score:
            best_score = score
            best_n = n
            
    return best_n
