import time
import math
from functools import lru_cache

def construct():
    start_time = time.time()
    timeout = 55.0
    
    target = math.factorial(12)  # 479001600
    
    # Prime factorization of 12!
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
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
    
    # We'll iterate M from smallest to largest
    # For each M, we try to find a subset of valid_divs[0..M_idx] that multiplies to target
    
    best_solution = None
    best_max = target + 1
    
    for M_idx in range(n_valid):
        M = valid_divs[M_idx]
        
        # Check if M alone could work (if M == target)
        if M == target:
            return [target]
        
        candidates = valid_divs[:M_idx+1]
        candidates_vec = valid_divs_vec[:M_idx+1]
        
        found_solution = None
        
        def dfs(start_idx, remaining_vec, count, current_factors):
            if time.time() - start_time > timeout:
                return None
            
            if all(v == 0 for v in remaining_vec):
                if count > 0:
                    return list(current_factors)
                return None
            
            if count >= 12:
                return None
            
            remaining_product = from_vector(remaining_vec)
            
            for i in range(start_idx, len(candidates)):
                c = candidates[i]
                cv = candidates_vec[i]
                
                # If c > remaining_product, break (since candidates are sorted)
                # This is a key pruning: we can't use factors larger than what's left
                if c > remaining_product:
                    break
                
                # Check if c divides remaining (using vector comparison)
                if can_divide(remaining_vec, cv):
                    current_factors.append(c)
                    new_remaining = subtract_vec(remaining_vec, cv)
                    res = dfs(i + 1, new_remaining, count + 1, current_factors)
                    if res:
                        current_factors.pop()
                        return res
                    current_factors.pop()
                
            return None

        target_vec = to_vector(target_pf)
        sol = dfs(0, target_vec, 0, [])
        
        if sol:
            # This is the first M we found that works, so it's optimal
            return sol
            
        if time.time() - start_time > timeout:
            break

    # Fallback: return a valid solution (not necessarily optimal)
    # Try a few more M values with a different search order
    for M_idx in range(n_valid - 1, -1, -1):
        M = valid_divs[M_idx]
        if M >= best_max:
            continue
        
        candidates = valid_divs[:M_idx+1]
        candidates_vec = valid_divs_vec[:M_idx+1]
        
        def dfs2(start_idx, remaining_vec, count, current_factors):
            if time.time() - start_time > timeout:
                return None
            
            if all(v == 0 for v in remaining_vec):
                if count > 0:
                    return list(current_factors)
                return None
            
            if count >= 12:
                return None
            
            remaining_product = from_vector(remaining_vec)
            
            # Try larger candidates first for this fallback
            for i in range(len(candidates) - 1, start_idx - 1, -1):
                if i < start_idx:
                    break
                c = candidates[i]
                cv = candidates_vec[i]
                
                if c > remaining_product:
                    continue
                
                if can_divide(remaining_vec, cv):
                    current_factors.append(c)
                    new_remaining = subtract_vec(remaining_vec, cv)
                    res = dfs2(i + 1, new_remaining, count + 1, current_factors)
                    if res:
                        current_factors.pop()
                        return res
                    current_factors.pop()
                
            return None

        target_vec = to_vector(target_pf)
        sol = dfs2(0, target_vec, 0, [])
        
        if sol:
            max_factor = max(sol)
            if max_factor < best_max:
                best_max = max_factor
                best_solution = sol
        
        if time.time() - start_time > timeout:
            break

    if best_solution:
        return best_solution
    
    # Final fallback: return a single factor
    return [target]
