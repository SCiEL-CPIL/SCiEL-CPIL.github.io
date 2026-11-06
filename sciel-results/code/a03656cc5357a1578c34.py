import time
import math
from functools import lru_cache

def construct():
    start_time = time.time()
    timeout = 55.0
    
    target = math.factorial(12)
    
    # Prime factorization of 12!
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    # Let's compute it properly
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
    
    # Precompute: for each index, the minimum product achievable with remaining slots
    # This is complex, so we'll use a different pruning strategy
    
    n_valid = len(valid_divs)
    
    # We'll iterate M from smallest to largest
    # For each M, we try to find a subset of valid_divs[0..M_idx] that multiplies to target
    
    # Use memoization on (start_idx, remaining_vec, count) -> found/failed
    # But remaining_vec can have many values, so let's be careful
    
    # Actually, let's just do DFS with strong pruning
    
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
            
            # Pruning: if the product of the smallest (12-count) available factors > remaining product, fail
            # But we need to be careful about distinctness and the start_idx constraint
            
            # Better pruning: 
            # 1. The remaining product must be >= the product of the smallest (12-count) factors from candidates[start_idx:]
            #    that we haven't used yet. But this is hard to compute quickly.
            
            # 2. If remaining product < minimum possible factor from candidates[start_idx:], fail
            #    (but we can always stop early, so this isn't quite right)
            
            # 3. Key insight: if the sum of exponents in remaining_vec is small, we're close to done
            
            # Let's try a different approach: bound the search by checking if the remaining product
            # can be factored into (12-count) distinct factors, all > 12, all <= M, all >= candidates[start_idx]
            
            # Simple check: if remaining product is 1, we're done (handled above)
            # If remaining product < candidates[start_idx] (if start_idx < len(candidates)), we can't pick any more factors
            # But we could also stop if count > 0 and remaining is 1
            
            # Actually, the key pruning is:
            # The maximum number of additional factors we can pick is 12 - count
            # The minimum product of (12-count) distinct factors from candidates[start_idx:] 
            # must be <= remaining product
            # But computing this exactly is expensive.
            
            # Alternative: just iterate and rely on the fact that we're trying M in increasing order,
            # so the first solution we find is optimal.
            
            remaining_product = from_vector(remaining_vec)
            
            for i in range(start_idx, len(candidates)):
                c = candidates[i]
                cv = candidates_vec[i]
                
                # If c > remaining_product, break (since candidates are sorted)
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
                
                # Pruning: if we've tried all factors that could divide remaining, and none worked,
                # we can break early in some cases, but it's complex.
                
            return None

        target_vec = to_vector(target_pf)
        sol = dfs(0, target_vec, 0, [])
        
        if sol:
            return sol
            
        if time.time() - start_time > timeout:
            break

    return [target]
