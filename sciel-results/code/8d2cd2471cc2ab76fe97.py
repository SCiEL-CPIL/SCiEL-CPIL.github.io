import sys
from math import factorial
from functools import lru_cache
import time

def construct():
    start_time = time.time()
    timeout = 55.0

    target = factorial(12)  # 479001600
    
    # Prime factorization: 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    # We'll use a DFS that builds factors one at a time
    # State: (current_factor_so_far, remaining_exponents, min_next_factor, max_factor_so_far)
    
    # Actually, let's use a cleaner approach:
    # We want to partition the prime factors into groups, each group forming a factor > 12
    # Each factor must be distinct
    
    # Let's use recursive search with the prime exponents as state
    # exponents: [e2, e3, e5, e7, e11]
    # At each step, we choose a factor f > 12 (and > all previously chosen factors to ensure distinctness and ordering)
    # f must divide the remaining product
    # We track the largest factor chosen so far
    
    # Better approach: generate all divisors > 12 of target, sort them, 
    # then use DFS with memoization on (index, remaining_product, max_factor)
    # But remaining_product can be large...
    
    # Let's use the prime exponent vector as state
    # State: (tuple of exponents, min_factor, max_factor)
    # We want to minimize max_factor
    
    # Actually, the cleanest approach:
    # 1. Get all divisors of target that are > 12
    # 2. Sort them
    # 3. DFS: at each step, pick a divisor from the remaining that divides the current product
    #    Track the max factor used
    #    Use memoization: (remaining_product_as_exponent_tuple, min_allowed_factor) -> min possible max_factor
    
    # Let me use exponent tuple representation
    # target = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
    # Represent a number by its exponent tuple (e2, e3, e5, e7, e11)
    
    def get_exp_tuple(n):
        """Get prime exponent tuple for n (assuming n divides target)"""
        e = [0, 0, 0, 0, 0]
        for i, p in enumerate([2, 3, 5, 7, 11]):
            while n % p == 0:
                e[i] += 1
                n //= p
        return tuple(e)
    
    target_exp = get_exp_tuple(target)  # (10, 5, 2, 1, 1)
    
    # Get all divisors of target that are > 12
    divisors = []
    for i in range(1, int(target ** 0.5) + 1):
        if target % i == 0:
            if i > 12:
                divisors.append(i)
            other = target // i
            if other > 12 and other != i:
                divisors.append(other)
    
    divisors = sorted(set(divisors))
    n_divs = len(divisors)
    
    # For each divisor, precompute its exponent tuple
    div_exp = {}
    for d in divisors:
        div_exp[d] = get_exp_tuple(d)
    
    # DFS with memoization
    # State: (exp_tuple, min_factor_idx, current_max)
    # We want to find if we can complete the factorization
    # And minimize the overall max factor
    
    # Better: DFS that returns the minimum possible max factor from this state
    # State: (exp_tuple, min_factor_value)
    # Returns: minimum possible max factor to complete, or infinity if impossible
    
    # But exp_tuple has 5 dimensions, each with limited range
    # e2: 0-10, e3: 0-5, e5: 0-2, e7: 0-1, e11: 0-1
    # Total states: 11*6*3*2*2 = 792
    # min_factor_value: up to n_divs choices
    # So total memo states: ~792 * n_divs which could be large
    
    # Let's limit: we only care about min_factor being one of the divisors
    # And we process divisors in order, so we can use an index instead
    
    # Alternative cleaner DFS:
    # Process divisors in sorted order. At each divisor, decide: use it or skip it.
    # State: (divisor_index, remaining_exp_tuple, max_factor_so_far)
    # But this is a subset selection problem which is NP-hard in general, 
    # but our state space is small enough.
    
    # Let's try: memo[(idx, exp_tuple)] = minimum possible max_factor from this state
    # where idx is the next divisor to consider (0-indexed)
    # We process divisors[0..idx-1] already decided, now at divisors[idx]
    
    INF = float('inf')
    
    @lru_cache(maxsize=None)
    def dfs(idx, e2, e3, e5, e7, e11, max_so_far):
        """
        Consider divisors[idx:]. 
        e2,e3,e5,e7,e11: remaining exponents to distribute.
        max_so_far: the maximum factor chosen so far.
        Returns: minimum possible final max factor, or INF if impossible.
        """
        # Check if we've used all prime factors
        if e2 == 0 and e3 == 0 and e5 == 0 and e7 == 0 and e11 == 0:
            return max_so_far
        
        # If no more divisors to consider
        if idx >= n_divs:
            return INF
        
        # Time check
        if time.time() - start_time > timeout:
            return INF
        
        best = INF
        
        # Option 1: Skip divisors[idx]
        result_skip = dfs(idx + 1, e2, e3, e5, e7, e11, max_so_far)
        if result_skip < best:
            best = result_skip
        
        # Option 2: Use divisors[idx] if it divides the remaining product
        d = divisors[idx]
        de = div_exp[d]
        de2, de3, de5, de7, de11 = de
        
        if e2 >= de2 and e3 >= de3 and e5 >= de5 and e7 >= de7 and e11 >= de11:
            new_max = max(max_so_far, d)
            # Pruning: if new_max is already >= best, no point
            if new_max < best:
                result_take = dfs(idx + 1, e2 - de2, e3 - de3, e5 - de5, e7 - de7, e11 - de11, new_max)
                if result_take < best:
                    best = result_take
        
        return best
    
    # Initial call: start at divisor index 0, full exponents, max_so_far = 12 (since all factors > 12)
    # Actually max_so_far should start at 0 or 12, doesn't matter since all factors > 12
    result = dfs(0, target_exp[0], target_exp[1], target_exp[2], target_exp[3], target_exp[4], 12)
    
    if result < INF:
        # Now reconstruct the solution
        # We need to re-run DFS but this time record the choices
        solution = []
        idx = 0
        e2, e3, e5, e7, e11 = target_exp
        max_so_far = 12
        
        while not (e2 == 0 and e3 == 0 and e5 == 0 and e7 == 0 and e11 == 0):
            if idx >= n_divs:
                break
            if time.time() - start_time > timeout:
                break
            
            d = divisors[idx]
            de = div_exp[d]
            de2, de3, de5, de7, de11 = de
            
            can_take = (e2 >= de2 and e3 >= de3 and e5 >= de5 and e7 >= de7 and e11 >= de11)
            
            # Check if skipping leads to a valid solution with max <= result
            skip_val = dfs(idx + 1, e2, e3, e5, e7, e11, max_so_far)
            
            if can_take:
                new_max = max(max_so_far, d)
                take_val = dfs(idx + 1, e2 - de2, e3 - de3, e5 - de5, e7 - de7, e11 - de11, new_max)
            else:
                take_val = INF
            
            # Decide: take if taking gives a better or equal result
            if take_val <= skip_val and take_val < INF:
                solution.append(d)
                e2 -= de2
                e3 -= de3
                e5 -= de5
                e7 -= de7
                e11 -= de11
                max_so_far = max(max_so_far, d)
            else:
                pass  # skip
            
            idx += 1
        
        if solution:
            return solution
    
    # Fallback
    return [target]

# EVOLVE-BLOCK-END
