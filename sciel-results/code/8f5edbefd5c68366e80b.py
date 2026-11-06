import math
import time
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 140  # Safety margin for 150s limit

    max_val = 10**18
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]
    
    # Precompute prime indices for bitmask operations
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    def factorize(n):
        """Factorize n and return list of (prime, exponent) pairs"""
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
        return factors

    def tau_perp_fast(n):
        """
        Fast tau_perp using bitmask representation.
        Each divisor is represented as a bitmask of which primes divide it.
        Two divisors are coprime iff their bitmasks are disjoint.
        """
        factors = factorize(n)
        if not factors:
            return 0
        
        # Map each prime in factors to a bit position
        prime_bits = {}
        for i, (p, e) in enumerate(factors):
            prime_bits[p] = i
        
        # Generate all divisors with their bitmask representations
        # divisor = product of p_i^(e_i) where 0 <= e_i <= exp_i
        # bitmask has bit i set if e_i > 0
        divs = []  # list of (divisor_value, bitmask)
        
        # Start with divisor 1, bitmask 0
        divs.append((1, 0))
        
        for i, (p, e) in enumerate(factors):
            current_len = len(divs)
            bit = 1 << i
            factor = 1
            for _ in range(e):
                factor *= p
                for j in range(current_len):
                    old_val, old_mask = divs[j]
                    divs.append((old_val * factor, old_mask | bit))
        
        # Sort by divisor value
        divs.sort(key=lambda x: x[0])
        
        # Count adjacent pairs with coprime bitmasks (disjoint)
        score = 0
        for i in range(len(divs) - 1):
            if (divs[i][1] & divs[i+1][1]) == 0:
                score += 1
        
        return score

    # Generate candidates using DFS with non-increasing exponents
    candidates = set()
    
    def dfs(idx, max_exp, current_n, current_tau):
        if time.time() - start_time > 40:  # Stop generating after 40s
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
            
            if time.time() - start_time > 40:
                return

    dfs(0, 60, 1, 1)
    
    # Add primorials and variations
    pk = 1
    for i, p in enumerate(primes):
        pk *= p
        if pk > max_val:
            break
        candidates.add((2**(i+1), pk))
        
        for a in range(1, 6):
            n = pk * (2 ** a)
            if n <= max_val:
                candidates.add((2**(i+1) * (a+1), n))
                
        for a in range(1, 5):
            n = pk * (3 ** a)
            if n <= max_val:
                candidates.add((2**(i+1) * (a+1), n))
                
        for a in range(1, 4):
            n = pk * (5 ** a)
            if n <= max_val:
                candidates.add((2**(i+1) * (a+1), n))

    if not candidates:
        return 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43 * 47

    # Convert to list and sort by tau descending
    candidate_list = list(candidates)
    candidate_list.sort(key=lambda x: x[0], reverse=True)
    
    best_score = -1
    best_n = candidate_list[0][1]
    
    # Evaluate top candidates with the fast method
    eval_limit = 20000  # Can evaluate more due to faster method
    evaluated = set()
    
    for i, (tau, n) in enumerate(candidate_list):
        if i >= eval_limit:
            break
        if time.time() - start_time > time_limit - 10:
            break
        if n in evaluated:
            continue
        evaluated.add(n)
        
        score = tau_perp_fast(n)
        if score > best_score:
            best_score = score
            best_n = n
            
    # Local search: try modifying the best candidate
    if time.time() - start_time < time_limit - 15:
        # Try swapping out one prime and adding another
        base_factors = factorize(best_n)
        base_n = best_n
        base_score = best_score
        
        # Try removing each prime and adding a different one
        for i, (p_remove, e_remove) in enumerate(base_factors):
            if time.time() - start_time > time_limit - 10:
                break
                
            # Remove this prime
            n_without = base_n // (p_remove ** e_remove)
            
            # Try adding different primes with various exponents
            for p_add in primes:
                if p_add == p_remove:
                    continue
                    
                # Try different exponents
                for e_add in range(1, 8):
                    if n_without * (p_add ** e_add) > max_val:
                        break
                        
                    new_n = n_without * (p_add ** e_add)
                    if new_n in evaluated:
                        continue
                    evaluated.add(new_n)
                    
                    score = tau_perp_fast(new_n)
                    if score > best_score:
                        best_score = score
                        best_n = new_n
        
        # Try adjusting exponents of existing primes
        for i, (p, e) in enumerate(base_factors):
            if time.time() - start_time > time_limit - 10:
                break
                
            # Try increasing exponent
            if e < 10:
                new_n = base_n * p
                if new_n <= max_val and new_n not in evaluated:
                    evaluated.add(new_n)
                    score = tau_perp_fast(new_n)
                    if score > best_score:
                        best_score = score
                        best_n = new_n
            
            # Try decreasing exponent
            if e > 1:
                new_n = base_n // p
                if new_n >= 2 and new_n not in evaluated:
                    evaluated.add(new_n)
                    score = tau_perp_fast(new_n)
                    if score > best_score:
                        best_score = score
                        best_n = new_n

    return best_n
