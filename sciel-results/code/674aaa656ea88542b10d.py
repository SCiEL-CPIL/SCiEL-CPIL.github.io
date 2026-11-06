"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    deadline = time.time() + 85
    target = Fraction(5, 31)
    MAX_DEN = 10000000
    MAX_TERMS = 256
    
    # Known solution: 1/8 + 1/28 + 1/1736 = 5/31
    best_max = 1736
    best_vals = [8, 28, 1736]
    
    def search_terms(k, min_d, used, current_vals, remaining, current_max):
        nonlocal best_max, best_vals
        if time.time() > deadline:
            return
        if remaining == 0:
            if len(current_vals) == k:
                if current_max < best_max:
                    best_max = current_max
                    best_vals = list(current_vals)
            return
        if remaining < 0:
            return
        if len(current_vals) >= k:
            return
        
        terms_left = k - len(current_vals)
        
        # Pruning: min possible sum with terms_left terms, each >= 1/best_max
        if remaining < Fraction(terms_left, best_max):
            return
        
        # Max possible sum: sum of reciprocals of smallest terms_left denominators from min_d
        max_possible = Fraction(0, 1)
        d = min_d
        count = 0
        while d <= best_max and count < terms_left:
            max_possible += Fraction(1, d)
            d += 1
            count += 1
        if remaining > max_possible:
            return
        
        # Next denominator bounds
        # 1/d <= remaining => d >= ceil(1/remaining)
        d_min_possible = max(min_d, (remaining.denominator + remaining.numerator - 1) // remaining.numerator)
        # terms_left * (1/d) >= remaining => d <= floor(terms_left / remaining)
        d_max_possible = min(best_max, terms_left * remaining.denominator // remaining.numerator)
        
        if d_min_possible > d_max_possible:
            return
        
        for d in range(d_min_possible, d_max_possible + 1):
            if time.time() > deadline:
                return
            if d in used:
                continue
            new_remaining = remaining - Fraction(1, d)
            if new_remaining < 0:
                break
            used.add(d)
            current_vals.append(d)
            search_terms(k, d + 1, used, current_vals, new_remaining, max(current_max, d))
            current_vals.pop()
            used.remove(d)
            if time.time() > deadline:
                return
    
    # Search for better 3-term solutions
    search_terms(3, 1, set(), [], target, 0)
    
    # Search for 4-term solutions if time remains
    if time.time() < deadline - 5:
        search_terms(4, 1, set(), [], target, 0)
    
    # Search for 5-term solutions if time remains
    if time.time() < deadline - 5:
        search_terms(5, 1, set(), [], target, 0)
    
    return best_vals
# EVOLVE-BLOCK-END
