import math
import time
import functools
import random

@functools.lru_cache(maxsize=100000)
def _score_from_factorization_cached(prime_tuple, exp_tuple):
    """
    Calculate tau_perp given the prime factorization.
    """
    primes = list(prime_tuple)
    exponents = list(exp_tuple)
    
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

def _score_from_factorization(primes, exponents):
    """
    Calculate tau_perp given the prime factorization.
    """
    return _score_from_factorization_cached(tuple(primes), tuple(exponents))

def construct():
    """
    Returns an integer n <= 10^18 maximizing tau_perp(n).
    """
    start_time = time.time()
    TIME_LIMIT = 115  # Safety margin for 120s limit
    
    max_val = 10**18
    
    # First 15 primes are sufficient for the base structures
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    best_score = 0
    best_n = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 # Seed fallback
    
    def try_structure(p_list, exps):
        nonlocal best_score, best_n
        val = 1
        for p, e in zip(p_list, exps):
            val *= (p ** e)
        if val <= max_val and val >= 2:
            score = _score_from_factorization(p_list, exps)
            if score > best_score:
                best_score = score
                best_n = val

    # 1. Evaluate square-free numbers (Primorials)
    # P_k is product of first k primes
    primorials = []
    prod = 1
    for i, p in enumerate(primes):
        prod *= p
        if prod > max_val:
            break
        primorials.append((i+1, prod))
        
    for k, p_val in primorials:
        try_structure(primes[:k], [1]*k)

    # 2. For each primorial P_k, try to multiply by powers of small primes
    # to use the remaining budget.
    # We focus on k=12, 13, 14, 15 as they have the most divisors.
    # For each, we define a "budget" B = max_val // P_k.
    # We search for multipliers m <= B composed of primes in primes[:k] (or even new primes? 
    # Adding a new prime increases divisor count significantly. 
    # If P_k uses first k primes, adding p_{k+1} might be possible if P_k * p_{k+1} <= max_val.
    # But P_{k+1} is already checked in step 1 if it fits.
    # So we only add powers of existing primes in P_k.
    
    for k, p_val in primorials:
        if time.time() - start_time > TIME_LIMIT:
            break
            
        budget = max_val // p_val
        if budget < 1:
            continue
            
        # We want to find exponents e_1, ..., e_k such that 
        # prod(p_i^(1+e_i)) <= max_val
        # Let delta_e_i be the extra exponent.
        # prod(p_i^delta_e_i) <= budget
        
        # Heuristic: The gain in score comes from adding divisors.
        # Adding a factor p_i^d adds d * (number of divisors of rest) new divisors?
        # Actually, it increases the total count.
        
        # We perform a local search / DFS on the exponents for the first few primes
        # because large primes have small budgets for exponents.
        
        # Determine which primes can have extra exponents.
        # If p_i > budget, delta_e_i must be 0.
        
        # We will DFS on the exponents of the first min(k, 8) primes, 
        # as exponents for larger primes will be 0 or 1 (if budget allows).
        
        def dfs_exp(idx, current_multiplier, current_exps):
            nonlocal best_score, best_n
            
            if time.time() - start_time > TIME_LIMIT:
                return

            # Evaluate current state
            # current_exps is the extra exponents for primes[:k]
            # Full exponents are [1 + e_1, 1 + e_2, ...]
            full_exps = [1 + e for e in current_exps]
            # Only consider primes that are in the base P_k
            # Note: current_exps corresponds to primes[:k]
            
            # Optimization: Only evaluate if we've changed something or at the end?
            # Evaluating is expensive. Let's prune.
            # If idx == k:
            #    Evaluate.
            
            if idx == k:
                # Check if multiplier is within budget
                if current_multiplier <= budget:
                    try_structure(primes[:k], full_exps)
                return

            p = primes[idx]
            
            # Max extra exponent for this prime
            # p^(1+e) <= budget / current_multiplier
            # e <= log_p(budget / current_multiplier) - 1
            if current_multiplier > budget:
                return
                
            max_e = 0
            if p <= budget // current_multiplier:
                # Find max e such that p^(e+1) <= budget // current_multiplier
                # Note: current_multiplier includes p^0 so far for this prime?
                # No, current_multiplier is the product of p_j^(delta_e_j) for j < idx.
                # We are deciding delta_e_idx.
                # The term contributed is p^(delta_e_idx).
                # So we need p^(delta_e_idx) <= budget // current_multiplier.
                
                limit = budget // current_multiplier
                e = 0
                while (p ** (e + 1)) <= limit:
                    e += 1
                max_e = e
            
            # Try all possible extra exponents from 0 to max_e
            # To improve speed, we might not need to try all if k is large.
            # But k is at most 15.
            # For small primes, max_e can be large.
            # For large primes, max_e is 0.
            
            for e in range(max_e + 1):
                new_mult = current_multiplier * (p ** e)
                if new_mult > budget:
                    break
                dfs_exp(idx + 1, new_mult, current_exps + [e])

        # Start DFS for this primorial base
        # We limit the depth of the DFS to the first few primes if k is large
        # to avoid combinatorial explosion.
        # Actually, for k=15, budget is small (~1.6), so max_e will be 0 for most.
        # For k=14, budget ~76.
        # For k=13, budget ~3289.
        
        # To ensure speed, we only DFS the first 10 primes.
        # For the rest, we assume delta_e = 0.
        # This is a heuristic.
        dfs_limit = min(k, 10)
        
        # Initialize extra exponents list
        # We pass a list of length k, but only fill the first dfs_limit.
        # The rest are 0.
        
        # Modify dfs_exp to handle the fixed tail
        def dfs_exp_limited(idx, current_multiplier, current_exps_partial):
            nonlocal best_score, best_n
            
            if time.time() - start_time > TIME_LIMIT:
                return

            if idx == dfs_limit:
                # Fill the rest with 0
                full_exps_partial = current_exps_partial + [0] * (k - dfs_limit)
                full_exps = [1 + e for e in full_exps_partial]
                
                # Check budget
                if current_multiplier <= budget:
                    try_structure(primes[:k], full_exps)
                return

            p = primes[idx]
            
            if current_multiplier > budget:
                return
                
            limit = budget // current_multiplier
            max_e = 0
            if p <= limit:
                e = 0
                while (p ** (e + 1)) <= limit:
                    e += 1
                max_e = e
            
            # Heuristic: Try larger exponents first? Or smaller?
            # Larger exponents increase divisor count more.
            # Let's try all.
            for e in range(max_e + 1):
                new_mult = current_multiplier * (p ** e)
                if new_mult > budget:
                    break
                dfs_exp_limited(idx + 1, new_mult, current_exps_partial + [e])

        dfs_exp_limited(0, 1, [])

    return best_n

# EVOLVE-BLOCK-END
