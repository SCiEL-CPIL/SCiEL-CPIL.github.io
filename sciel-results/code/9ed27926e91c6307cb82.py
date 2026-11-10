"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from functools import lru_cache

def construct():
    """
    Find 1..12 DISTINCT integers all greater than 12, with product exactly 12!.
    Minimize the LARGEST factor.
    """
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    target = math.factorial(12)  # 479001600
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    
    # Generate all divisors of 12! that are > 12
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    primes = [2, 3, 5, 7, 11]
    exponents = [10, 5, 2, 1, 1]
    
    # Generate all divisors using prime factorization
    divisors = [1]
    for p, e in zip(primes, exponents):
        new_divs = []
        for d in divisors:
            val = d
            for _ in range(e + 1):
                new_divs.append(val)
                val *= p
        divisors = new_divs
    
    # Filter: must be > 12 and < target (proper divisors, since we need at least 1 factor)
    # Actually, we need distinct factors > 12. The single factor case is [target].
    # For multiple factors, each factor is a proper divisor.
    # Let's keep all divisors > 12.
    divisors = [d for d in divisors if d > 12]
    divisors.sort()
    
    # Precompute prime exponent vectors for each divisor
    # This allows us to use a "remaining exponents" state in DFS, which is more robust
    # than just "remaining product" when combined with set-based pruning.
    # However, "remaining product" is sufficient if we check divisibility.
    # The key improvement from inspirations:
    # 1. Better pruning using the fact that factors must be distinct and increasing.
    # 2. Iterating only through valid divisors of the current remaining product.
    
    # To make iteration fast, we can precompute divisors of each possible remaining product?
    # No, remaining product changes.
    # Instead, we iterate through the global sorted list of divisors, but skip those
    # that don't divide the current remaining product.
    
    # Optimization: Since the list of divisors is small (~700), a linear scan with modulo check is fine.
    # But we can be smarter: if we maintain the set of prime exponents remaining,
    # we can check if a candidate divisor's exponent vector is <= remaining exponents.
    
    # Let's stick to the product-based DFS but with tighter pruning.
    # Key insight from inspiration 2: 
    # "The next factor d must satisfy: d > last_factor"
    # And we iterate through sorted divisors.
    
    best_solution = None
    best_max = float('inf')
    
    # To speed up, we can precompute the index of each divisor in the sorted list
    # and use binary search to start from the correct index.
    # But simple iteration is likely fast enough given the small size.
    
    def dfs(remaining, min_factor, used_set, current_max, depth):
        nonlocal best_solution, best_max, deadline, start_time
        
        # Time check
        if time.time() - start_time > 85:
            return False
        
        # If remaining is 1, we have a complete solution
        if remaining == 1:
            if depth >= 1:
                if current_max < best_max:
                    best_max = current_max
                    best_solution = list(used_set)
            return True
        
        # Pruning: if current_max >= best_max, no point
        if current_max >= best_max:
            return False
        
        # Pruning: if we have used 12 factors, we can't add more
        if depth >= 12:
            return False
        
        # If remaining is a valid factor itself ( > min_factor, not used, > 12 )
        # we can take it as the last factor.
        # This is handled in the loop below if we allow d == remaining.
        
        # Iterate through possible next factors d
        # d must be:
        # 1. A divisor of 'remaining'
        # 2. d >= min_factor
        # 3. d not in used_set
        # 4. d > 12 (guaranteed by our divisor list)
        # 5. max(current_max, d) < best_max
        
        # Optimization: 
        # We can break early if d >= best_max (since divisors are sorted, larger d will also be >= best_max)
        # Wait, if d >= best_max, then new_max >= best_max, so we can break.
        
        for d in divisors:
            if d < min_factor:
                continue
            
            # If d is too large to improve best_max, we can break
            # Because divisors are sorted, any subsequent d' >= d will also be >= best_max
            # Note: We need new_max = max(current_max, d) < best_max
            # So if d >= best_max, we can break.
            if d >= best_max:
                break
            
            if d > remaining:
                break
            
            if remaining % d != 0:
                continue
            
            if d in used_set:
                continue
            
            new_remaining = remaining // d
            new_max = max(current_max, d)
            
            # Pruning: new_max >= best_max (already checked by d < best_max, but current_max might be larger? No, current_max < best_max is checked at top)
            # Actually, if current_max is close to best_max, and d is small, new_max might still be < best_max.
            # The check `if d >= best_max: break` is safe because if d >= best_max, new_max >= d >= best_max.
            
            # Pruning: new_remaining must be expressible with factors > d
            # If new_remaining > 1, the smallest possible next factor is d+1.
            # So we need new_remaining >= d+1? Not exactly, new_remaining is a product.
            # But if new_remaining is prime and < d+1, it's impossible.
            # More generally, if new_remaining < d+1 and new_remaining != 1, it's impossible.
            if new_remaining > 1 and new_remaining < d + 1:
                continue
            
            # Additional pruning: 
            # The number of factors we can still add is 12 - depth - 1 (after adding d)
            # The minimum product of k distinct integers > d is (d+1)(d+2)...(d+k)
            # If new_remaining < (d+1), it's impossible to have even 1 factor.
            # If new_remaining < (d+1)(d+2), it's impossible to have 2 factors.
            # We don't know how many factors are left, but we know it's at least 1 (if new_remaining != 1).
            
            # Let's add the factor and recurse
            used_set.add(d)
            dfs(new_remaining, d + 1, used_set, new_max, depth + 1)
            used_set.discard(d)
            
            # Early termination: if we found a solution with max <= d, 
            # no need to try larger d
            if best_max <= d:
                break
        
        return best_solution is not None
    
    # Start the DFS
    dfs(target, 13, set(), 0, 0)
    
    if best_solution is not None:
        return best_solution
    else:
        # Fallback: return [12!] (single factor)
        return [target]

# EVOLVE-BLOCK-END
