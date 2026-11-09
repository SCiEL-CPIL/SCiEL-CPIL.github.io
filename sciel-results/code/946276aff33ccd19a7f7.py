import time
from math import factorial, gcd
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin before 60s limit
    
    N = factorial(12)  # 479001600
    
    # Known good solution: max factor 64
    best_candidate = [22, 21, 25, 18, 36, 64]
    best_max = 64
    
    # Try to find solutions with smaller max factor
    # Strategy: try max values from 13 upward, search for valid decompositions
    
    # Precompute prime factorization of N
    # 12! = 2^10 * 3^5 * 5^2 * 7 * 11
    prime_powers = {2: 10, 3: 5, 5: 2, 7: 1, 11: 1}
    
    def get_prime_factors(n):
        """Get prime factorization as dict."""
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
    
    def can_divide(remaining, factor):
        """Check if factor divides remaining (as prime factor counts)."""
        pf = get_prime_factors(factor)
        for p, exp in pf.items():
            if remaining.get(p, 0) < exp:
                return False
        return True
    
    def subtract_factors(remaining, factor):
        """Subtract factor's prime exponents from remaining."""
        pf = get_prime_factors(factor)
        new_rem = dict(remaining)
        for p, exp in pf.items():
            new_rem[p] = new_rem.get(p, 0) - exp
            if new_rem[p] == 0:
                del new_rem[p]
        return new_rem
    
    # Backtracking search with max_factor constraint
    def search(remaining, min_factor, max_factor_allowed, current_factors, max_depth=12):
        """
        Search for a valid decomposition.
        remaining: dict of prime -> exponent still needed
        min_factor: minimum value for next factor (to ensure distinct and increasing)
        max_factor_allowed: upper bound on any factor
        current_factors: list of factors chosen so far
        Returns list of factors if found, None otherwise.
        """
        if time.time() > deadline:
            return None
        
        # If remaining is empty, we've found a valid decomposition
        if not remaining:
            if len(current_factors) >= 1:
                return current_factors[:]
            return None
        
        # If we've used too many factors
        if len(current_factors) >= max_depth:
            return None
        
        # Calculate the product of remaining primes
        rem_product = 1
        for p, exp in remaining.items():
            rem_product *= (p ** exp)
        
        # If rem_product is 1, done (should be caught above)
        if rem_product == 1:
            return current_factors[:] if current_factors else None
        
        # The next factor must be at least min_factor and at most max_factor_allowed
        # Also, the next factor must divide rem_product
        # And if this is the last factor, it must equal rem_product
        
        start = max(min_factor, 13)  # factors must be > 12
        end = min(max_factor_allowed, rem_product)
        
        # If start > end, no valid factor in range
        if start > end:
            return None
        
        # Try each possible next factor
        for f in range(start, end + 1):
            if rem_product % f != 0:
                continue
            
            if not can_divide(remaining, f):
                continue
            
            new_remaining = subtract_factors(remaining, f)
            new_factors = current_factors + [f]
            
            # Pruning: if new_remaining is not empty, the product of remaining
            # must be achievable with factors all > f and <= max_factor_allowed
            if new_remaining:
                next_rem_product = 1
                for p, exp in new_remaining.items():
                    next_rem_product *= (p ** exp)
                
                # The smallest possible next factor is f+1
                # We need to be able to decompose next_rem_product into factors in [f+1, max_factor_allowed]
                # Quick check: next_rem_product must be >= (f+1)
                if next_rem_product < f + 1:
                    continue
                
                # Also check that next_rem_product can be formed
                # Simple bound: if next_rem_product > max_factor_allowed and we can't split it, fail
                # This is a weak check, but helps pruning
                
                result = search(new_remaining, f + 1, max_factor_allowed, new_factors, max_depth)
                if result is not None:
                    return result
            else:
                # Found a complete solution
                return new_factors
        
        return None
    
    # Try to find solutions with progressively smaller max factors
    # Start from 63 and go down
    for max_f in range(63, 12, -1):
        if time.time() > deadline:
            break
        
        # Reset for each max_f attempt
        result = search(prime_powers, 13, max_f, [])
        if result is not None:
            # Verify the solution
            prod = 1
            for c in result:
                prod *= c
            if prod == N and len(result) <= 12 and all(c > 12 for c in result) and len(set(result)) == len(result):
                best_candidate = result
                best_max = max_f
                break
    
    # Return the best solution found
    return best_candidate
