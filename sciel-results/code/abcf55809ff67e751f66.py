import math
import time
from itertools import product

def construct():
    start_time = time.time()
    N = 479001600  # 12!
    
    # We need to find k distinct integers > 12 such that their product is N.
    # Minimize the largest factor.
    
    # 1. Prime factorization of N
    # 12! = 479001600
    # Primes <= 12: 2, 3, 5, 7, 11
    # 2^10 * 3^5 * 5^2 * 7 * 11
    # 1024 * 243 * 25 * 7 * 11 = 479001600
    
    prime_powers = [
        (2, 10),
        (3, 5),
        (5, 2),
        (7, 1),
        (11, 1)
    ]
    
    # Generate all divisors of N
    divisors = [1]
    for p, e in prime_powers:
        new_divs = []
        p_power = 1
        for i in range(e + 1):
            for d in divisors:
                new_divs.append(d * p_power)
            p_power *= p
        divisors = new_divs
    
    # Filter divisors > 12
    valid_divs = sorted([d for d in divisors if d > 12])
    
    # Precompute a set for O(1) lookup
    valid_divs_set = set(valid_divs)
    
    best_max_val = N
    best_factors = [N]
    
    def check_time():
        return time.time() - start_time > 55
    
    # We will search for k from 7 down to 2.
    # k=1 is [N], which is the initial best.
    # k > 7 is impossible because 13*14*15*16*17*18*19*20 > N.
    
    def dfs(target_k, current_prod, last_factor, depth, path):
        nonlocal best_max_val, best_factors
        
        if check_time():
            return
            
        if depth == target_k:
            if current_prod == N:
                max_f = max(path)
                if max_f < best_max_val:
                    best_max_val = max_f
                    best_factors = list(path)
            return
        
        remaining = target_k - depth
        target_rem = N // current_prod
        
        # Pruning: Minimum product of remaining distinct factors > last_factor
        min_prod = 1
        for i in range(1, remaining + 1):
            min_prod *= (last_factor + i)
            
        if target_rem < min_prod:
            return
            
        # If remaining == 1, the factor is fixed
        if remaining == 1:
            x = target_rem
            if x > last_factor and x > 12 and x in valid_divs_set:
                path.append(x)
                dfs(target_k, current_prod * x, x, depth + 1, path)
                path.pop()
            return
        
        # Determine range for next factor x
        lower_bound = max(last_factor + 1, 13)
        
        # Upper bound estimation:
        # x * (x+1) * ... * (x+remaining-1) <= target_rem
        # x^remaining <= target_rem  => x <= target_rem^(1/remaining)
        # Add a small margin to be safe
        x_upper = int(target_rem ** (1.0 / remaining)) + 10
        
        # Iterate through valid divisors in range
        # We can iterate through valid_divs (sorted) and break when > x_upper
        for x in valid_divs:
            if x < lower_bound:
                continue
            if x > x_upper:
                break
                
            if target_rem % x != 0:
                continue
                
            # Additional pruning: 
            # The remaining product is target_rem // x
            # It must be divisible by (x+1)...? No, just distinct and > x.
            # But the next factors must divide target_rem // x.
            
            path.append(x)
            dfs(target_k, current_prod * x, x, depth + 1, path)
            path.pop()
            
            if check_time():
                return

    # Search k=7 down to 2
    # We can also try to prune k if the minimum possible max factor for k
    # is already >= best_max_val.
    # Min max factor for k distinct factors > 12 with product N:
    # Ideally they are close to N^(1/k).
    # But distinctness pushes them apart.
    # A rough lower bound on the max factor is N^(1/k).
    # If N^(1/k) >= best_max_val, then no solution with k factors can improve best_max_val?
    # Not necessarily, because we minimize the MAX.
    # If N^(1/k) is small, the factors are small.
    # Wait, if N^(1/k) is large, the factors are large.
    # We want to MINIMIZE the max.
    # So if the theoretical minimum max for k factors is >= current best, we can skip k?
    # The theoretical minimum max for k factors is at least N^(1/k).
    # So if N^(1/k) >= best_max_val, then any solution with k factors has max >= N^(1/k) >= best_max_val.
    # So we can skip k.
    
    for k in range(7, 1, -1):
        # Check if it's worth searching k
        # N^(1/k) is the geometric mean.
        # If the geometric mean is already >= best_max_val, we can't do better.
        # Actually, the max factor is at least the geometric mean.
        # So if N^(1/k) >= best_max_val, skip.
        if N ** (1.0 / k) >= best_max_val:
            continue
            
        dfs(k, 1, 12, 0, [])
        
        if check_time():
            break

    return best_factors

# EVOLVE-BLOCK-START
# The above code is the logic. I will wrap it in the required format.
# EVOLVE-BLOCK-END
