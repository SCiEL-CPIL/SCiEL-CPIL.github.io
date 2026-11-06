"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    target = Fraction(5, 31)
    
    # Known 3-term solution: 1/8 + 1/31 + 1/248 = 5/31
    # Max denominator is 248.
    best_solution = [8, 31, 248]
    best_max = 248
    
    def try_update(denoms):
        nonlocal best_max, best_solution
        m = max(denoms)
        if m < best_max:
            # Verify sum exactly
            s = sum(Fraction(1, d) for d in denoms)
            if s == target:
                best_max = m
                best_solution = list(denoms)
                return True
        return False

    def search(terms, rem_val, min_denom, current):
        nonlocal best_max, best_solution
        if time.time() > deadline:
            return False
        
        if terms == 0:
            return rem_val == 0
        
        if terms == 1:
            n, d = rem_val.numerator, rem_val.denominator
            if n == 0:
                return False
            if d % n != 0:
                return False
            denom = d // n
            if denom > min_denom and denom < best_max:
                if try_update(current + [denom]):
                    return True
            return False
        
        # Pruning:
        # 1. We need 'terms' distinct denominators > min_denom and < best_max.
        #    So we need at least 'terms' integers in (min_denom, best_max).
        #    Count of such integers is best_max - 1 - min_denom.
        if best_max - 1 - min_denom < terms:
            return False
        
        # 2. Max possible sum with 'terms' denominators > min_denom:
        #    Use min_denom+1, min_denom+2, ..., min_denom+terms
        max_sum = sum(Fraction(1, min_denom + i) for i in range(1, terms + 1))
        if max_sum < rem_val:
            return False
        
        # 3. Min possible sum with 'terms' denominators < best_max:
        #    Use best_max-1, best_max-2, ..., best_max-terms
        #    Note: These must also be > min_denom. If not, the range is invalid, handled above.
        min_sum = sum(Fraction(1, best_max - i) for i in range(1, terms + 1))
        if min_sum > rem_val:
            return False
        
        # Determine range for the next denominator 'a'
        # 1/a < rem_val => a > 1/rem_val
        # terms/a > rem_val => a < terms/rem_val
        a_min = max(min_denom + 1, int(1 / rem_val) + 1)
        a_max = min(int(terms / rem_val), best_max - 1)
        
        if a_min > a_max:
            return False
        
        for a in range(a_min, a_max + 1):
            new_rem = rem_val - Fraction(1, a)
            if new_rem < 0:
                break
            if search(terms - 1, new_rem, a, current + [a]):
                return True
        return False

    # Search for 4-term solutions with max < 248
    search(4, target, 6, [])
    if best_max < 248:
        return best_solution

    # Search for 5-term solutions with max < best_max
    search(5, target, 6, [])
    if best_max < 248:
        return best_solution

    # Search for 6-term solutions with max < best_max
    search(6, target, 6, [])
    
    return best_solution

# EVOLVE-BLOCK-END
