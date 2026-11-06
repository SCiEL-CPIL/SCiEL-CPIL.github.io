import math
import time
from itertools import product

T12 = math.factorial(12)
# 12! = 2^10 * 3^5 * 5^2 * 7^1 * 11^1
PRIMES = [2, 3, 5, 7, 11]
TOTAL_EXP = [10, 5, 2, 1, 1]

def exp_to_int(e):
    r = 1
    for i, p in enumerate(PRIMES):
        r *= p ** e[i]
    return r

def valid_solution(factors):
    if not (1 <= len(factors) <= 12):
        return False
    if len(set(factors)) != len(factors):
        return False
    if any(f < 13 for f in factors):
        return False
    p = 1
    for f in factors:
        p *= f
    return p == T12

def construct():
    deadline = time.time() + 85
    best_max = T12
    best_factors = [T12]
    
    # Precompute all divisors of 12! that are > 12 and < 12!
    # Store as (value, exp_vector)
    all_divisors = []
    for e2 in range(11):
        for e3 in range(6):
            for e5 in range(3):
                for e7 in range(2):
                    for e11 in range(2):
                        v = (2**e2)*(3**e3)*(5**e5)*(7**e7)*(11**e11)
                        if v > 12 and v < T12:
                            all_divisors.append((v, [e2, e3, e5, e7, e11]))
    all_divisors.sort()
    
    # Build a lookup from value to exp vector
    div_lookup = {v: e for v, e in all_divisors}
    div_values = [v for v, e in all_divisors]
    
    # Try to find a solution with maximum factor <= M
    # Binary search on M or try increasing M from a lower bound
    # Lower bound: at least 13 (smallest factor > 12)
    # Upper bound: T12
    
    def can_decompose(max_factor, rem_exp, min_val, count, factors, used_set):
        """Try to decompose rem_exp into factors in (min_val, max_factor], distinct, count more factors."""
        nonlocal best_max, best_factors
        
        if time.time() > deadline:
            return False
        
        # Check if rem_exp is zero
        if all(x == 0 for x in rem_exp):
            if count >= 1:
                cur_max = max(factors)
                if cur_max < best_max:
                    best_max = cur_max
                    best_factors = factors[:]
                return True
            return False
        
        if count >= 12:
            return False
        
        # The remaining product
        rem_prod = 1
        for i, p in enumerate(PRIMES):
            rem_prod *= p ** rem_exp[i]
        
        # If rem_prod <= min_val, can't form a factor > min_val
        if rem_prod <= min_val:
            return False
        
        # If rem_prod > max_factor and count == 0, we need at least 2 factors
        # The smallest possible next factor is min_val + 1
        # If rem_prod / (min_val + 1) > max_factor and we need more than 1 factor, prune
        # Actually, let's think: we need to pick a factor f where min_val < f <= max_factor
        # and f divides rem_prod, and the remainder can also be decomposed
        
        # Find all divisors of rem_prod that are in (min_val, max_factor]
        # and have exponent vectors that are sub-vectors of rem_exp
        candidates = []
        for v, e in all_divisors:
            if v <= min_val or v > max_factor:
                break
            # Check if e is a sub-vector of rem_exp
            valid = True
            for i in range(5):
                if e[i] > rem_exp[i]:
                    valid = False
                    break
            if valid and v not in used_set:
                candidates.append((v, e))
        
        if not candidates:
            return False
        
        # Try each candidate, sorted by value
        for v, e in candidates:
            if time.time() > deadline:
                return False
            
            new_rem = [rem_exp[i] - e[i] for i in range(5)]
            
            # If new_rem is all zeros, this is the last factor
            if all(x == 0 for x in new_rem):
                cur_max = max(factors + [v])
                if cur_max < best_max:
                    best_max = cur_max
                    best_factors = factors + [v]
                return True
            
            # Recurse
            if can_decompose(max_factor, new_rem, v, count + 1, factors + [v], used_set | {v}):
                return True
        
        return False
    
    # Try increasing maximum factor bounds
    # Start from a reasonable lower bound
    # The product is 12! ≈ 479M. With up to 12 factors each > 12, 
    # the minimum possible max is at least (12!)^(1/12) ≈ 13.2
    # Let's try bounds from 20 upward
    
    # First, try some hand-crafted solutions
    test_solutions = [
        [13, 14, 15, 16, 18, 20, 21, 22, 24, 27, 30, 33],  # too many, check
        [14, 15, 16, 18, 20, 21, 22, 24, 25, 27, 30, 33],
        [13, 14, 15, 16, 18, 20, 21, 22, 24, 25, 27, 30],
        [14, 15, 16, 18, 20, 21, 22, 24, 25, 27, 30, 32],
        [13, 14, 15, 16, 18, 20, 21, 22, 24, 25, 27, 32],
        [14, 15, 16, 18, 20, 21, 22, 24, 25, 27, 32, 33],
        [13, 14, 15, 16, 18, 20, 21, 22, 24, 25, 32, 33],
        [14, 15, 16, 18, 20, 21, 22, 24, 25, 32, 33, 35],
        [13, 14, 15, 16, 18, 20, 21, 22, 24, 25, 33, 35],
        [14, 15, 16, 18, 20, 21, 22, 24, 32, 33, 35, 36],
    ]
    for sol in test_solutions:
        if valid_solution(sol):
            if max(sol) < best_max:
                best_max = max(sol)
                best_factors = sol[:]
    
    # Systematic search: try max_factor from lower to upper
    # Lower bound estimate: we need product = 12!, with factors > 12
    # If all factors are around M, then M^12 >= 12! => M >= (12!)^(1/12) ≈ 13.2
    # But with distinct factors, the minimum max is higher
    
    # Try max_factor values
    for M in range(20, 200):
        if time.time() > deadline - 5:
            break
        
        # Try to decompose with max factor = M
        # Use backtracking
        found = can_decompose(M, TOTAL_EXP, 12, 0, [], set())
        if found:
            # Found a solution with max <= M
            # Since we're going in increasing order, this is optimal for this M
            # But we should check if we can do better with a smaller M
            # Actually, we're iterating M from low to high, so the first M that works is optimal
            break
    
    if best_factors and valid_solution(best_factors):
        return best_factors
    return [T12]
