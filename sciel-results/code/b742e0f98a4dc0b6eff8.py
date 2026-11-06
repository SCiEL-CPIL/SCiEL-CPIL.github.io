"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import sys
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    N = 12
    target = math.factorial(N)  # 479001600
    min_factor = N + 1  # 13
    
    # Precompute all divisors of target for quick reference
    # target = 2^10 * 3^5 * 5^2 * 7 * 11
    # Number of divisors: (10+1)*(5+1)*(2+1)*(1+1)*(1+1) = 11*6*3*2*2 = 792
    
    def get_all_divisors(n):
        divs = []
        for i in range(1, int(math.isqrt(n)) + 1):
            if n % i == 0:
                divs.append(i)
                if i != n // i:
                    divs.append(n // i)
        return sorted(divs)
    
    all_divs_of_target = get_all_divisors(target)
    # Filter to only those > 12
    valid_divs = [d for d in all_divs_of_target if d > 12]
    
    # Get an initial upper bound by trying a greedy approach
    # Try to find a decomposition with moderate factors
    def greedy_decompose():
        """Try to find a reasonable decomposition greedily."""
        remaining = target
        factors = []
        current_min = min_factor
        
        while remaining > 1:
            # Find the largest divisor of remaining that is >= current_min
            # and as small as possible to keep factors balanced
            best_d = None
            for d in valid_divs:
                if d >= current_min and remaining % d == 0:
                    if best_d is None or d < best_d:
                        best_d = d
                    if best_d > remaining // (current_min + 1):
                        break
            if best_d is None:
                # Try larger divisors
                for d in valid_divs:
                    if d >= current_min and remaining % d == 0:
                        best_d = d
                        break
            if best_d is None:
                break
            factors.append(best_d)
            remaining //= best_d
            current_min = best_d + 1
            if len(factors) >= 12:
                break
        
        if remaining == 1 and len(factors) >= 1:
            return max(factors), factors
        return None, None
    
    # Try greedy to get initial bound
    greedy_max, greedy_factors = greedy_decompose()
    
    # Also try some specific known good decompositions
    # 12! = 479001600
    # Let's try to find a good one by checking specific factor sets
    candidate_sets = [
        # Various combinations to try
    ]
    
    best_largest = [target]
    best_factors = [target]
    
    if greedy_max is not None and greedy_max < best_largest[0]:
        best_largest[0] = greedy_max
        best_factors[:] = greedy_factors[:]
    
    # Additional heuristic: try to balance factors
    # For k factors, the optimal max is when all are ~ (target)^(1/k)
    # target^(1/3) ≈ 782, target^(1/4) ≈ 262, target^(1/5) ≈ 137, target^(1/6) ≈ 89
    # So with 5-6 factors, we might get max around 100-200
    
    # Use a smarter DFS with memoization on (remaining, min_allowed)
    # to avoid re-exploring the same states
    
    # Precompute: for each possible remaining value (divisor of target),
    # store the divisors that are > some threshold
    
    divisor_cache = {}
    
    def get_divisors_above(n, threshold):
        key = (n, threshold)
        if key in divisor_cache:
            return divisor_cache[key]
        
        # Get all divisors of n
        if n not in divisor_cache:
            divs = get_all_divisors(n)
            divisor_cache[n] = divs
        else:
            divs = divisor_cache[n]
        
        # Filter to those >= threshold and < best_largest
        result = [d for d in divs if d >= threshold and d < best_largest[0]]
        divisor_cache[key] = result
        return result
    
    # Stronger pruning: check if remaining can be factored into distinct integers
    # all >= min_allowed, using only primes that divide remaining
    def can_factor(remaining, min_allowed, max_factors):
        """Check if remaining can be written as product of 1..max_factors distinct integers all >= min_allowed."""
        if remaining == 1:
            return True
        if max_factors <= 0:
            return False
        if remaining < min_allowed:
            return False
        
        # Quick check: if remaining is prime and remaining < min_allowed, fail
        # If remaining is prime, it must be one of the factors
        if remaining >= min_allowed:
            # remaining itself could be a factor
            return True  # optimistic: assume it's possible
        
        # Check if remaining can be split
        # Try smallest factor first
        for d in range(min_allowed, int(math.isqrt(remaining)) + 1):
            if remaining % d == 0:
                quotient = remaining // d
                if quotient >= d + 1:  # next factor must be > d
                    if can_factor(quotient, d + 1, max_factors - 1):
                        return True
        
        return False
    
    # Even stronger: use prime factorization to bound
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
    
    def dfs(remaining, min_allowed, factors_so_far, current_max):
        if time.time() > deadline:
            return
        
        # Prune: if current_max already >= best, no improvement possible
        if current_max >= best_largest[0]:
            return
        
        # Base case
        if remaining == 1:
            if factors_so_far:
                if current_max < best_largest[0]:
                    best_largest[0] = current_max
                    best_factors[:] = factors_so_far[:]
            return
        
        # If we've used 12 factors already and remaining != 1, dead end
        if len(factors_so_far) >= 12:
            return
        
        # Quick prune: remaining must be >= min_allowed (at least one more factor needed)
        if remaining < min_allowed:
            return
        
        # Stronger prune: check if remaining can possibly be factored
        # into distinct factors all >= min_allowed
        # The minimum number of factors needed: at least 1
        # The maximum number: 12 - len(factors_so_far)
        max_factors_left = 12 - len(factors_so_far)
        
        # Check: can remaining be formed by 1..max_factors_left distinct factors >= min_allowed?
        # Quick necessary condition: remaining must have enough "prime mass"
        # For now, use a lighter check
        
        # Find divisors of remaining that are >= min_allowed and < best_largest
        divs = get_divisors_above(remaining, min_allowed)
        
        if not divs:
            return
        
        for d in divs:
            quotient = remaining // d
            
            # If quotient == 1, we're done
            if quotient == 1:
                new_max = max(current_max, d)
                if new_max < best_largest[0]:
                    best_largest[0] = new_max
                    best_factors[:] = factors_so_far + [d]
                continue
            
            # Lower bound check: quotient must be >= d+1 (next factor must be > d)
            if quotient < d + 1:
                continue
            
            # Additional prune: quotient must be factorable into distinct factors all > d
            # Quick check: if quotient is prime and quotient <= d, fail
            # If quotient is prime, it must be > d
            if quotient > 1:
                # Check if quotient can be formed by distinct factors all >= d+1
                # Light check: quotient must be >= d+1 (already checked)
                # Also, if quotient is prime, it must be >= d+1
                pass
            
            new_max = max(current_max, d)
            if new_max >= best_largest[0]:
                continue
            
            factors_so_far.append(d)
            dfs(quotient, d + 1, factors_so_far, new_max)
            factors_so_far.pop()
    
    # Start DFS
    dfs(target, min_factor, [], 0)
    
    return best_factors[:]
# EVOLVE-BLOCK-END
