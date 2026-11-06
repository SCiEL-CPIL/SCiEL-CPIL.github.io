"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random

def construct():
    target = math.factorial(12)
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Get all divisors of 12!
    def get_all_divisors(n):
        divs = set()
        for i in range(1, int(math.isqrt(n)) + 1):
            if n % i == 0:
                divs.add(i)
                divs.add(n // i)
        return sorted(divs)
    
    all_divs = get_all_divisors(target)
    # Filter to divisors > 12
    valid_divs = [d for d in all_divs if d > 12]
    valid_divs.sort()
    
    best_factors = None
    best_max = float('inf')
    
    # 1. Try to find a good initial solution using a heuristic
    # We want to partition the prime factors of 12! into K groups
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    primes = [2]*10 + [3]*5 + [5]*2 + [7] + [11]
    
    def try_greedy_partition(k):
        # Try to partition primes into k bins with balanced products
        # Sort primes descending to place large ones first
        sorted_primes = sorted(primes, reverse=True)
        bins = [1] * k
        for p in sorted_primes:
            # Find bin with min product
            min_idx = 0
            min_val = bins[0]
            for j in range(1, k):
                if bins[j] < min_val:
                    min_val = bins[j]
                    min_idx = j
            bins[min_idx] *= p
        
        # Check validity: all > 12 and distinct
        if any(b <= 12 for b in bins):
            return None
        if len(set(bins)) != k:
            return None
        
        return bins

    # Try different numbers of factors k
    # Theoretical min k is around 4-5, max is 7
    for k in range(4, 8):
        sol = try_greedy_partition(k)
        if sol:
            if max(sol) < best_max:
                best_max = max(sol)
                best_factors = sol
    
    # If no greedy solution found, or to improve it, run DFS
    # We can limit the DFS time or use the best_max as a tight bound
    
    def dfs(idx, remaining, current_max, current_factors):
        nonlocal best_factors, best_max
        
        # Time check
        if time.time() > deadline:
            return
        
        if remaining == 1:
            # Found a valid decomposition
            if current_max < best_max:
                best_max = current_max
                best_factors = list(current_factors)
            return
        
        if remaining <= 12:
            # Can't use any more factors since all must be > 12
            return
        
        # Pruning: if current_max >= best_max, no need to continue
        if current_max >= best_max:
            return
            
        # Additional pruning: 
        # If we have picked `len(current_factors)` factors, we can pick at most 12 - len(current_factors) more.
        # The maximum possible product we can form with the remaining factors is bounded.
        # But more importantly, the MINIMUM product of the remaining factors (if we need more)
        # must be <= remaining.
        
        # If remaining > 1, we need at least one more factor.
        # The smallest possible factor is valid_divs[idx] (if idx < len(valid_divs)).
        # If valid_divs[idx] > remaining, we can't pick any more factors (since they are sorted).
        # But wait, we are in the loop below which handles this.
        
        # Let's add a check: if remaining is not divisible by any valid_divs[i] for i >= idx, return.
        # This is expensive to check every time.
        
        # Heuristic: if remaining is prime or has no small factors, it might be hard.
        
        for i in range(idx, len(valid_divs)):
            d = valid_divs[i]
            
            if d > remaining:
                break
            
            if remaining % d != 0:
                continue
            
            new_max = max(current_max, d)
            if new_max >= best_max:
                # Since valid_divs is sorted, if d >= best_max, all subsequent d' >= d >= best_max
                # so we can break.
                if d >= best_max:
                    break
                # If d < best_max but new_max >= best_max (meaning current_max >= best_max? No, current_max < best_max)
                # This implies d >= best_max? No.
                # If current_max < best_max and d < best_max, then new_max < best_max.
                # So this condition `new_max >= best_max` only triggers if d >= best_max.
                # So the break is correct.
            
            current_factors.append(d)
            dfs(i + 1, remaining // d, new_max, current_factors)
            current_factors.pop()
            
            if time.time() > deadline:
                return
    
    # Start DFS
    # We can start with the best_max from greedy to prune heavily
    dfs(0, target, 0, [])
    
    if best_factors is not None:
        return best_factors
    
    # Fallback
    return [target]

# EVOLVE-BLOCK-END
