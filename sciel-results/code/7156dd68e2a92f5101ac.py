import math
import time
from functools import lru_cache

def construct():
    """
    Returns an integer n <= 10^18 maximizing the number of adjacent
    coprime pairs in its divisor sequence.
    """
    start_time = time.time()
    TIME_LIMIT = 140  # Safety margin below 150s
    
    max_val = 10**18
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    def tau_perp(n):
        """Compute the score: number of adjacent coprime pairs in sorted divisors."""
        if n <= 1:
            return 0
        # Factorize n
        factors = []
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                factors.append((p, count))
        if temp > 1:
            factors.append((temp, 1))
        
        # Generate all divisors
        divs = [1]
        for p, exp in factors:
            current_size = len(divs)
            factor = 1
            for _ in range(exp):
                factor *= p
                for i in range(current_size):
                    divs.append(divs[i] * factor)
        
        divs.sort()
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score
    
    # Strategy: 
    # 1. Try square-free numbers with many primes
    # 2. Try numbers with small exponents on many primes
    # 3. Use DFS with pruning
    
    candidates = []
    
    # Add square-free candidates
    sf_n = 1
    for p in primes:
        if sf_n * p <= max_val:
            sf_n *= p
        else:
            break
    candidates.append(sf_n)
    
    # Also try removing some large primes and adding smaller ones with higher exponents
    # DFS approach: allow e=0 to skip primes
    def dfs(idx, max_exp, current_n, current_tau, current_factors):
        if time.time() - start_time > TIME_LIMIT:
            return
        
        if current_n > 1:
            candidates.append(current_n)
        
        if idx >= len(primes):
            return
        
        p = primes[idx]
        limit_n = max_val // current_n
        
        # Calculate max exponent
        e_max_possible = 0
        temp = 1
        while temp * p <= limit_n and e_max_possible < max_exp:
            temp *= p
            e_max_possible += 1
        
        # Try e from 0 to e_max_possible
        # But to keep search manageable, try e=0,1,2,3 and up to e_max for small primes
        max_e_to_try = min(e_max_possible, max(3, 60 // (idx + 1))) if idx < 5 else min(e_max_possible, 2)
        
        for e in range(0, max_e_to_try + 1):
            if e == 0:
                next_n = current_n
                next_tau = current_tau
                next_factors = current_factors
            else:
                next_n = current_n * (p ** e)
                next_tau = current_tau * (e + 1)
                next_factors = current_factors + [(p, e)]
            
            dfs(idx + 1, e if e > 0 else max_exp, next_n, next_tau, next_factors)
            
            if time.time() - start_time > TIME_LIMIT:
                return
    
    # Start DFS - but this could explode. Let's be more targeted.
    # Instead, generate specific candidate families:
    
    # Family 1: Square-free with first k primes
    sf_n = 1
    for p in primes:
        if sf_n * p <= max_val:
            candidates.append(sf_n * p)
            sf_n *= p
        else:
            break
    
    # Family 2: 2^a * 3^b * 5^c * ... with small exponents
    # Try all combinations where exponents are 1 or 2 for first ~15 primes
    # This is still too many. Let's use a smarter approach.
    
    # Key insight: The score is roughly equal to the number of "prime transitions" in the divisor sequence.
    # For a number with k distinct prime factors, the minimum score is k (from 1, p1, p2, ..., pk).
    # To get more, we need the sorted divisors to have more coprime adjacent pairs.
    
    # Let's try a different approach: generate candidates with many distinct primes
    # and evaluate them.
    
    # Generate: product of first k primes for k = 1 to max
    k_max = 0
    prod = 1
    for p in primes:
        if prod * p <= max_val:
            prod *= p
            k_max += 1
            candidates.append(prod)
        else:
            break
    
    # Generate: 2^a * product of next k primes, for small a
    base = 1
    for p in primes[1:]:  # skip 2
        if base * p <= max_val // 8:  # leave room for 2^3
            base *= p
        else:
            break
    
    for a in range(1, 10):
        if (2 ** a) * base <= max_val:
            candidates.append((2 ** a) * base)
    
    # Generate: 2^a * 3^b * product of next primes
    base2 = 1
    for p in primes[2:]:  # skip 2,3
        if base2 * p <= max_val // 36:  # leave room for 2^2 * 3^2
            base2 *= p
        else:
            break
    
    for a in range(1, 6):
        for b in range(1, 6):
            if (2 ** a) * (3 ** b) * base2 <= max_val:
                candidates.append((2 ** a) * (3 ** b) * base2)
    
    # Deduplicate
    candidates = list(set(candidates))
    
    # Filter: only keep n >= 2
    candidates = [n for n in candidates if 2 <= n <= max_val]
    
    # Sort by number of distinct prime factors (heuristic: more primes = potentially more coprime pairs)
    def num_distinct_primes(n):
        temp = n
        count = 0
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                count += 1
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            count += 1
        return count
    
    candidates.sort(key=num_distinct_primes, reverse=True)
    
    # Evaluate top candidates
    best_n = candidates[0] if candidates else 2
    best_score = -1
    
    # Limit evaluation to stay within time
    eval_limit = 2000
    for n in candidates[:eval_limit]:
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        score = tau_perp(n)
        if score > best_score:
            best_score = score
            best_n = n
    
    return best_n
