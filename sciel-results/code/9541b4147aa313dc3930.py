"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    def solve(remaining, min_d, chosen, max_terms, time_limit):
        if time.time() > time_limit:
            return None
        if remaining == 0:
            return chosen[:]
        if max_terms == 0 or remaining < 0:
            return None
        
        # Pruning
        if remaining > Fraction(max_terms, min_d):
            return None
        if remaining < Fraction(max_terms, 10000000):
            return None
        
        d_min = min_d
        if remaining < 1:
            d_min = max(d_min, (remaining.denominator + remaining.numerator - 1) // remaining.numerator)
        
        d_cap = min(10000000, max(d_min + 200000, 2 * remaining.denominator // remaining.numerator if remaining.numerator > 0 else 10000000))
        
        for d in range(d_min, d_cap + 1):
            if time.time() > time_limit:
                return None
            term = Fraction(1, d)
            if term > remaining:
                break
            new_rem = remaining - term
            if new_rem > Fraction(max_terms - 1, d + 1):
                continue
            if new_rem < Fraction(max_terms - 1, 10000000):
                break
            
            chosen.append(d)
            result = solve(new_rem, d + 1, chosen, max_terms - 1, time_limit)
            chosen.pop()
            if result is not None:
                return result
        return None
    
    time_per_attempt = 10
    for n_terms in range(8, 14):
        t_start = time.time()
        t_limit = min(deadline, t_start + time_per_attempt)
        result = solve(Fraction(1, 1), 5, [], n_terms, t_limit)
        if result is not None:
            s = sum(Fraction(1, d) for d in result)
            if s == 1 and all(5 <= d <= 10000000 for d in result) and len(set(result)) == len(result):
                return result
    
    # Fallback
    target, reached = 2520, {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
    
    raise RuntimeError("no representation found")
# EVOLVE-BLOCK-END
