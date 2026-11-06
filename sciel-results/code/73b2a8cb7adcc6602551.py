"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from math import factorial, gcd, prod
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    N = factorial(12)  # 479001600
    MIN_VAL = 13  # all factors must be > 12
    MAX_FACTORS = 12
    
    # Generate all divisors of N
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    primes = [2, 3, 5, 7, 11]
    exponents = [10, 5, 2, 1, 1]
    
    divisors = [1]
    for p, e in zip(primes, exponents):
        new_divisors = []
        for d in divisors:
            for i in range(e + 1):
                new_divisors.append(d * (p ** i))
        divisors = new_divisors
    
    # Filter divisors > 12
    valid_divisors = sorted([d for d in divisors if d > 12])
    
    # For efficient lookup
    divisor_set = set(valid_divisors)
    
    best_solution = None
    best_max = N  # single factor case [N]
    
    # Precompute prime factorization exponents for N
    # N = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    prime_exp_N = {2: 10, 3: 5, 5: 2, 7: 1, 11: 1}
    
    def factorize(n):
        """Get prime factorization of n as a dict."""
        result = {}
        temp = n
        for p in primes:
            while temp % p == 0:
                result[p] = result.get(p, 0) + 1
                temp //= p
        return result
    
    # Precompute prime factorization for all valid divisors
    divisor_factorizations = {}
    for d in valid_divisors:
        divisor_factorizations[d] = factorize(d)
    
    # Use a recursive search with pruning
    def dfs(remaining, min_idx, current_factors, depth, max_depth, current_max, rem_exp):
        nonlocal best_solution, best_max
        nonlocal deadline
        
        if time.time() > deadline:
            return
        
        if depth == max_depth:
            if remaining == 1:
                if len(current_factors) == max_depth:
                    max_f = max(current_factors)
                    if max_f < best_max:
                        best_max = max_f
                        best_solution = current_factors[:]
            return
        
        if remaining == 1:
            # We have fewer factors than max_depth, but that's fine
            # Actually we need exactly max_depth factors
            return
        
        # Pruning: if current_max is already >= best_max, no improvement possible
        m_remaining = max_depth - depth
        if m_remaining > 0:
            min_possible_max = int(remaining ** (1.0 / m_remaining)) + 1
            if current_max >= best_max and min_possible_max >= best_max:
                return
        
        # Upper bound on next factor
        max_f_possible = int(remaining ** (1.0 / m_remaining)) + 2
        
        # Iterate through valid_divisors from min_idx
        for i in range(min_idx, len(valid_divisors)):
            if time.time() > deadline:
                return
            
            f = valid_divisors[i]
            
            if f > max_f_possible:
                break
            
            if remaining % f != 0:
                continue
            
            if f in current_factors:
                continue
            
            # Check prime exponent feasibility
            f_factors = divisor_factorizations[f]
            new_rem_exp = dict(rem_exp)
            feasible = True
            for p, e in f_factors.items():
                if p in new_rem_exp:
                    new_rem_exp[p] -= e
                    if new_rem_exp[p] < 0:
                        feasible = False
                        break
                else:
                    feasible = False
                    break
            if not feasible:
                continue
            
            # Additional pruning: remaining/f must be factorable into (m_remaining-1) factors > f
            if m_remaining - 1 > 0:
                min_prod = 1
                for j in range(1, m_remaining):
                    min_prod *= (f + j)
                    if min_prod > remaining // f:
                        break
                if min_prod > remaining // f:
                    break
            
            new_factors = current_factors + [f]
            dfs(remaining // f, i + 1, new_factors, depth + 1, max_depth, max(current_max, f), new_rem_exp)
    
    # Try different numbers of factors, starting with more (likely to give smaller max)
    for k in range(MAX_FACTORS, 1, -1):
        if time.time() > deadline:
            break
        
        # Lower bound check: can we even have k factors?
        min_possible = 1
        for i in range(k):
            min_possible *= (13 + i)
        
        if min_possible > N:
            continue
        
        # Theoretical minimum max for k factors is at least N^(1/k)
        lower_bound = int(N ** (1.0 / k)) + 1
        
        # If we already have a solution with max <= lower_bound, we can't improve for this k
        if best_solution is not None and best_max <= lower_bound:
            break
        
        # Start the DFS with k factors
        dfs(N, 0, [], 0, k, 0, dict(prime_exp_N))
        
        # Early termination if we found a very good solution
        if best_solution and best_max <= lower_bound + 2:
            break
    
    if best_solution is None:
        return [N]
    
    return best_solution

# EVOLVE-BLOCK-END
