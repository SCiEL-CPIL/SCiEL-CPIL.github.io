"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import gcd

def _greedy_complete(r, start_d, max_d, used):
    """Complete r = sum(1/d_i) with d_i >= start_d, all distinct, not in used.
    Returns list of denominators or None."""
    result = []
    while r > 0:
        # Smallest d >= start_d such that 1/d <= r
        d = max(start_d, (r.denominator + r.numerator - 1) // r.numerator)
        if d in used:
            d += 1
        if d > max_d:
            return None
        result.append(d)
        used.add(d)
        r -= Fraction(1, d)
        start_d = d + 1
    return result

def _dfs_egyptian(target, max_d, max_terms, used, deadline, best_max):
    """DFS to find Egyptian fraction for target with denominators <= max_d,
    at most max_terms terms, distinct, not in used.
    Returns (list_of_denoms, max_denom) or (None, None)."""
    if target <= 0:
        return [], 0
    
    # Lower bound on number of terms needed
    # If target > 0, need at least 1 term
    # Upper bound: each term is at most 1/(min_possible_d)
    
    # Smallest possible next denominator
    min_d = 1
    for d in used:
        if d >= min_d:
            min_d = d + 1
    # Actually we need d > all used? No, just distinct.
    # But for DFS efficiency, let's enforce increasing order
    # Let's track the minimum allowed denominator
    
    # For increasing order: next_d must be > all previously chosen
    if used:
        min_d = max(used) + 1
    else:
        min_d = 1
    
    # If 1/min_d < target, we can't reach target with denominators >= min_d
    # Actually we need sum of 1/d_i = target, each d_i >= min_d
    # Max possible sum with k terms is k / min_d
    # Min possible sum is... well, depends
    
    # Pruning: if target > max_terms / min_d, impossible (each term <= 1/min_d)
    if max_terms > 0 and target > Fraction(max_terms, min_d):
        return None, None
    
    # Pruning: if target < 1/(max_d * max_terms), impossible (each term >= 1/max_d, so min sum with max_terms terms is... no)
    # Actually min sum with 1 term is 1/max_d. If target < 1/max_d and we need at least 1 term, impossible.
    # But we could use more terms. Min sum with t terms where each <= max_d is... 1/max_d for each, so t/max_d.
    # If target < 1/max_d, we still need at least 1 term, and the smallest single term we can use is 1/max_d.
    # So if target < 1/max_d, impossible with any number of terms (since all terms are positive and each >= 1/max_d... wait no, each term is 1/d where d <= max_d, so each term >= 1/max_d).
    # So min possible value of any single term is 1/max_d. If target < 1/max_d, impossible.
    if target < Fraction(1, max_d):
        return None, None
    
    # Try each possible next denominator
    # d must satisfy: 1/d <= target (so d >= ceil(1/target))
    # and d >= min_d, d <= max_d
    d_start = max(min_d, (target.denominator + target.numerator - 1) // target.numerator)
    d_end = max_d
    
    # Also, if we use this term, remaining = target - 1/d must be achievable
    # with (max_terms - 1) terms, each >= d+1, each <= max_d
    
    for d in range(d_start, d_end + 1):
        if time.time() > deadline:
            return None, None
        if d in used:
            continue
        
        rem = target - Fraction(1, d)
        if rem < 0:
            continue
        
        if rem == 0:
            # Found a solution
            candidate_max = d
            if candidate_max < best_max[0]:
                best_max[0] = candidate_max
                return [d], d
            continue
        
        # Check if rem is achievable
        # rem must be >= 1/max_d (if we need at least 1 more term)
        # and rem <= (max_terms - 1) / (d + 1) (max sum with remaining terms)
        if max_terms - 1 > 0:
            if rem < Fraction(1, max_d):
                continue
            if rem > Fraction(max_terms - 1, d + 1):
                continue
        
        new_used = used | {d}
        result, mx = _dfs_egyptian(rem, max_d, max_terms - 1, new_used, deadline, best_max)
        if result is not None:
            full = [d] + result
            full_max = max(d, mx)
            if full_max < best_max[0]:
                best_max[0] = full_max
                return full, full_max
    
    return None, None

def construct():
    start_time = time.time()
    deadline = start_time + 75  # Safety margin before 90s limit
    
    target = Fraction(5, 31)
    
    # First, get a baseline solution using greedy
    r = target
    greedy_result = []
    used = set()
    while r > 0:
        d = (r.denominator + r.numerator - 1) // r.numerator
        if d in used:
            d += 1
        greedy_result.append(d)
        used.add(d)
        r -= Fraction(1, d)
    
    best_max = [max(greedy_result)]
    best_solution = greedy_result[:]
    
    # Now try to find better solutions with DFS
    # Try with small max_d values, increasing
    
    # The minimum possible max denominator is at least ceil(1/target) = ceil(31/5) = 7
    # But we need sum of reciprocals = 5/31 ≈ 0.1613
    # With 1 term: need 1/d = 5/31, so d = 31/5, not integer
    # With 2 terms: 1/a + 1/b = 5/31
    #   (31a + 31b) / (31ab) = 5/31 => 31(a+b) = 5ab => 5ab - 31a - 31b = 0
    #   (5a - 31)(5b - 31) = 31*31 = 961
    #   961 = 1 * 961 = 31 * 31
    #   Case 1: 5a-31=1, 5b-31=961 => a=32/5 (not int)
    #   Case 2: 5a-31=31, 5b-31=31 => a=62/5 (not int)
    #   So no 2-term solution with integer denominators.
    
    # With 3 terms: 1/a + 1/b + 1/c = 5/31
    # Let's try to find 3-term solutions
    
    # For 3 terms, assume a < b < c
    # 1/a + 1/b + 1/c = 5/31
    # a must satisfy: 1/a < 5/31 < 3/a => a > 31/5 = 6.2 and a < 3*31/5 = 18.6
    # So a in {7, 8, ..., 18}
    
    found_better = False
    for a in range(7, 19):
        if time.time() > deadline:
            break
        rem2 = target - Fraction(1, a)
        if rem2 <= 0:
            continue
        # 1/b + 1/c = rem2, with b > a, c > b
        # b must satisfy: 1/b < rem2 < 2/b => b > 1/rem2 and b < 2/rem2
        # Also b > a
        b_start = max(a + 1, (rem2.denominator + rem2.numerator - 1) // rem2.numerator + 1)
        # b < 2/rem2 => b < 2*rem2.denominator / rem2.numerator
        b_end = int(2 * rem2.denominator / rem2.numerator)
        
        for b in range(b_start, b_end + 1):
            if time.time() > deadline:
                break
            rem3 = rem2 - Fraction(1, b)
            if rem3 <= 0:
                continue
            # c = 1/rem3
            if rem3.denominator % rem3.numerator == 0:
                c = rem3.denominator // rem3.numerator
                if c > b:
                    max_d = c
                    if max_d < best_max[0]:
                        best_max[0] = max_d
                        best_solution = [a, b, c]
                        found_better = True
    
    # Try 4-term solutions if time permits
    if time.time() < deadline - 5:
        for a in range(7, 19):
            if time.time() > deadline - 5:
                break
            rem2 = target - Fraction(1, a)
            if rem2 <= 0:
                continue
            b_start = max(a + 1, (rem2.denominator + rem2.numerator - 1) // rem2.numerator + 1)
            b_end = min(int(2 * rem2.denominator / rem2.numerator), 100)
            
            for b in range(b_start, b_end + 1):
                if time.time() > deadline - 5:
                    break
                rem3 = rem2 - Fraction(1, b)
                if rem3 <= 0:
                    continue
                # 1/c + 1/d = rem3, c > b, d > c
                c_start = max(b + 1, (rem3.denominator + rem3.numerator - 1) // rem3.numerator + 1)
                c_end = min(int(2 * rem3.denominator / rem3.numerator), 200)
                
                for c in range(c_start, c_end + 1):
                    if time.time() > deadline - 5:
                        break
                    rem4 = rem3 - Fraction(1, c)
                    if rem4 <= 0:
                        continue
                    if rem4.denominator % rem4.numerator == 0:
                        d = rem4.denominator // rem4.numerator
                        if d > c:
                            max_d = d
                            if max_d < best_max[0]:
                                best_max[0] = max_d
                                best_solution = [a, b, c, d]
    
    return best_solution
# EVOLVE-BLOCK-END
