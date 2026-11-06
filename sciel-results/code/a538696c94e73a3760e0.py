import time
import math

def construct():
    start_time = time.time()
    timeout = 55.0
    
    target = math.factorial(12)
    
    # Prime factorization of 12!
    def prime_factorization(n):
        factors = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        return factors

    target_pf = prime_factorization(target)
    primes = sorted(target_pf.keys())
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    def to_vector(pf):
        vec = [0] * len(primes)
        for p, exp in pf.items():
            vec[prime_to_idx[p]] = exp
        return tuple(vec)
    
    def from_vector(vec):
        result = 1
        for i, p in enumerate(primes):
            result *= p ** vec[i]
        return result

    def can_divide(remaining_vec, factor_vec):
        for i in range(len(primes)):
            if remaining_vec[i] < factor_vec[i]:
                return False
        return True
    
    def subtract_vec(remaining_vec, factor_vec):
        return tuple(remaining_vec[i] - factor_vec[i] for i in range(len(primes)))

    # Generate all divisors of target
    divisors = []
    for i in range(1, int(math.isqrt(target)) + 1):
        if target % i == 0:
            divisors.append(i)
            if i != target // i:
                divisors.append(target // i)
    
    # Filter divisors > 12 and precompute their prime exponent vectors
    valid_divs = sorted([d for d in divisors if d > 12])
    valid_divs_vec = [to_vector(prime_factorization(d)) for d in valid_divs]
    
    n_valid = len(valid_divs)
    target_vec = to_vector(target_pf)
    
    # We iterate M from smallest to largest.
    # For each M, we try to find a subset of distinct divisors in (12, M] 
    # that multiplies to target.
    # To avoid permutations, we enforce strictly increasing order of factors.
    
    for M_idx in range(n_valid):
        M = valid_divs[M_idx]
        
        # Candidates are valid_divs[0 .. M_idx]
        # We perform a DFS to find an increasing sequence of distinct candidates
        # that multiplies to target.
        
        found_solution = None
        
        def dfs(start_idx, remaining_vec, current_factors):
            if time.time() - start_time > timeout:
                return None
            
            if all(v == 0 for v in remaining_vec):
                if current_factors:
                    return list(current_factors)
                return None
            
            # Heuristic: If remaining_product is 1, we are done (handled above)
            # If remaining_product > 1, we need at least one more factor.
            
            remaining_product = from_vector(remaining_vec)
            
            # If the smallest possible next factor (start_idx) is too large, 
            # or if we can't reach remaining_product with available factors?
            
            # We iterate from start_idx to M_idx
            # We enforce order: next factor must be > previous factor.
            # Since we start at 0 and increment index, this is handled by start_idx.
            
            for i in range(start_idx, M_idx + 1):
                c = valid_divs[i]
                cv = valid_divs_vec[i]
                
                # Pruning: if c > remaining_product, no larger c will work either
                if c > remaining_product:
                    break
                
                if can_divide(remaining_vec, cv):
                    current_factors.append(c)
                    new_remaining = subtract_vec(remaining_vec, cv)
                    res = dfs(i + 1, new_remaining, current_factors)
                    if res:
                        current_factors.pop()
                        return res
                    current_factors.pop()
                
            return None

        sol = dfs(0, target_vec, [])
        
        if sol:
            return sol

        if time.time() - start_time > timeout:
            break

    # Fallback if no solution found (should not happen)
    return [target]
