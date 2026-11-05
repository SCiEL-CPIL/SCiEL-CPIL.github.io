"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import factorial
from functools import lru_cache

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill

    target = factorial(12)  # 479001600

    # Prime factorization of 12!
    # 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    # We'll work with exponent vectors: (e2, e3, e5, e7, e11)
    target_exp = (10, 5, 2, 1, 1)

    # Precompute all divisors of 12! that are > 12
    # and store their exponent vectors
    divisors_exp = {}  # value -> (e2, e3, e5, e7, e11)
    divisors_sorted = []  # sorted list of (value, exp_tuple)
    
    # Generate all divisors by iterating through exponent combinations
    for e2 in range(11):
        for e3 in range(6):
            for e5 in range(3):
                for e7 in range(2):
                    for e11 in range(2):
                        val = (2**e2) * (3**e3) * (5**e5) * (7**e7) * (11**e11)
                        if val > 12:
                            exp_tuple = (e2, e3, e5, e7, e11)
                            divisors_exp[val] = exp_tuple
                            divisors_sorted.append((val, exp_tuple))
    
    divisors_sorted.sort()
    max_divisor = divisors_sorted[-1][0]

    # We want to find a set of distinct divisors > 12 whose product is 12!
    # and minimize the largest factor.
    
    # Key insight: We can binary search on the maximum factor M.
    # For a given M, check if we can form 12! using distinct factors in (12, M].
    # This is a set packing / exact cover problem on exponent vectors.
    
    # Alternative approach: DFS with strong pruning, searching for solutions
    # with decreasing maximum factor.
    
    # Let's use a smarter approach:
    # 1. Start with a known upper bound (e.g., the trivial 2-factor solution)
    # 2. Use DFS with pruning to find solutions with smaller max factors
    # 3. Try different numbers of factors k
    
    # First, find a good initial upper bound
    # Trivial: [13, 12!//13] -> max = 12!//13 = 36846277
    # Better: try to find balanced decompositions
    
    # Let's enumerate possible factor counts k from 1 to 8
    # (max k is limited because min product of k distinct factors > 12 grows quickly)
    
    # Min product of k distinct factors > 12:
    # k=1: 13
    # k=2: 13*14 = 182
    # k=3: 13*14*15 = 2730
    # k=4: 13*14*15*16 = 43680
    # k=5: 13*14*15*16*17 = 742560
    # k=6: 13*14*15*16*17*18 = 13366080
    # k=7: 13*14*15*16*17*18*19 = 253955520
    # k=8: 13*14*15*16*17*18*19*20 = 5079110400 > 12!
    # So k can be at most 7 (since 5079110400 > 479001600)
    # Wait, let me recalculate:
    # 13*14*15*16*17*18*19 = ?
    # 13*14=182, *15=2730, *16=43680, *17=742560, *18=13366080, *19=253955520
    # 253955520 < 479001600, so k=7 is possible
    # 13*14*15*16*17*18*19*20 = 253955520*20 = 5079110400 > 479001600
    # So max k = 7
    
    best_max = float('inf')
    best_factors = None

    # Helper: check if remaining exponent vector can be formed by k_left distinct factors
    # from the set of allowed divisors (all > 12, not in used, and < current best_max)
    # This is expensive, so we use it sparingly.

    def dfs(remaining_exp, used_set, current_factors, current_max, k_left, min_factor):
        nonlocal best_max, best_factors, start_time
        
        if time.time() - start_time > DEADLINE:
            return
        
        if k_left == 0:
            if remaining_exp == (0, 0, 0, 0, 0):
                if current_max < best_max:
                    best_max = current_max
                    best_factors = list(current_factors)
            return
        
        # If remaining_exp is all zeros but k_left > 0, invalid
        if all(e == 0 for e in remaining_exp):
            return
        
        # Pruning: if current_max >= best_max, stop
        if current_max >= best_max:
            return
        
        # Pruning: the minimum possible product of k_left distinct factors >= min_factor
        # that are divisors of the remaining number must be <= the remaining number.
        # But we work with exponents, so let's think differently.
        
        # The remaining number is determined by remaining_exp.
        # remaining_num = 2^e2 * 3^e3 * 5^e5 * 7^e7 * 11^e11
        remaining_num = (2**remaining_exp[0]) * (3**remaining_exp[1]) * (5**remaining_exp[2]) * (7**remaining_exp[3]) * (11**remaining_exp[4])
        
        # Quick check: if remaining_num < min_factor**k_left (rough), we can't form k_left factors
        # Actually, the min product of k_left distinct factors >= min_factor is at least
        # min_factor * (min_factor+1) * ... * (min_factor+k_left-1)
        # But these may not divide remaining_num. Let's use a weaker bound.
        if remaining_num < min_factor**k_left:
            return
        
        # The largest factor in the remaining k_left factors must be at least remaining_num**(1/k_left)
        if remaining_num**(1.0/k_left) >= best_max:
            return
        
        # Try each possible next factor
        # We need d such that:
        # - d > 12, d < best_max
        # - d not in used_set
        # - d >= min_factor (to avoid permutations)
        # - d divides remaining_num
        # - the exponent vector of d is <= remaining_exp component-wise
        
        # Iterate through divisors_sorted, but only those with value in [min_factor, best_max)
        # and not in used_set
        
        for d_val, d_exp in divisors_sorted:
            if d_val < min_factor:
                continue
            if d_val >= best_max:
                break
            if d_val in used_set:
                continue
            
            # Check if d_exp <= remaining_exp
            if any(d_exp[i] > remaining_exp[i] for i in range(5)):
                continue
            
            # Compute new remaining_exp
            new_remaining_exp = tuple(remaining_exp[i] - d_exp[i] for i in range(5))
            
            new_max = max(current_max, d_val)
            if new_max >= best_max:
                continue
            
            dfs(new_remaining_exp, used_set | {d_val}, current_factors + [d_val], new_max, k_left - 1, d_val + 1)
    
    # Search for solutions with k factors, for k from 7 down to 1
    # For each k, try to find a solution with minimal max factor.
    # We can use iterative deepening on the max factor.
    
    # First, let's try a direct DFS for each k, trying to find any solution.
    # Then we can refine.
    
    for k in range(7, 0, -1):
        if time.time() - start_time > DEADLINE:
            break
        
        # For this k, do DFS to find a solution
        # Start with min_factor = 13
        dfs(target_exp, set(), [], 0, k, 13)
        
        if best_factors is not None:
            # Found a solution with k factors
            # We could try to improve by reducing the max factor further
            # But for now, let's continue to other k values to see if we find better
            pass
    
    if best_factors is None:
        # Fallback
        return [13, target // 13]
    
    return best_factors
# EVOLVE-BLOCK-END
