from fractions import Fraction
from itertools import combinations

def construct():
    # Try to find minimum number of distinct denominators >= 5 with reciprocal sum = 1
    target = Fraction(1, 1)
    
    for n in range(8, 257):
        # Check if it's even possible: max sum with n terms starting from 5
        max_sum = sum(Fraction(1, i) for i in range(5, 5 + n))
        if max_sum < target:
            continue
        
        # Use backtracking search
        result = search(target, 5, n, [])
        if result is not None:
            return result
    
    raise RuntimeError("no representation found")

def search(remaining, min_denom, terms_left, chosen):
    if terms_left == 0:
        return chosen if remaining == 0 else None
    
    if remaining <= 0:
        return None
    
    # Pruning: if even taking the smallest possible denominators can't reach remaining
    # Max possible sum with terms_left terms starting from min_denom
    max_possible = sum(Fraction(1, i) for i in range(min_denom, min_denom + terms_left))
    if max_possible < remaining:
        return None
    
    # Min possible sum with terms_left terms (using very large denominators)
    # If remaining is too large to be achieved... actually min sum approaches 0, so this isn't a great prune
    
    # Try each possible next denominator
    # The next denominator d must satisfy: 1/d <= remaining (so d >= 1/remaining)
    # And d >= min_denom
    # Also, we need to be able to achieve remaining - 1/d with terms_left - 1 terms, all > d
    
    # Upper bound on d: if 1/d > remaining, skip. So d >= ceil(1/remaining)
    # But also, the remaining sum after choosing d must be achievable with terms_left-1 terms all > d
    # Max sum with terms_left-1 terms all > d is sum(1/i for i in range(d+1, d+terms_left))
    
    start_d = max(min_denom, (1 / remaining).numerator // (1 / remaining).denominator + 1) if remaining < 1 else min_denom
    # Actually, 1/d <= remaining means d >= 1/remaining
    if remaining < 1:
        start_d = max(min_denom, (1 / remaining).numerator // (1 / remaining).denominator)
        # Need d such that 1/d <= remaining, i.e., d >= 1/remaining
        # 1/remaining = remaining.denominator / remaining.numerator
        min_d_from_sum = (remaining.denominator + remaining.numerator - 1) // remaining.numerator  # ceil(1/remaining)
        start_d = max(min_denom, min_d_from_sum)
    else:
        start_d = min_denom
    
    # Also, we need at least terms_left denominators available from start_d onwards
    # And the max sum with terms_left terms from start_d must be >= remaining
    # Already checked above
    
    # Upper bound: even if we take the smallest possible remaining terms, we need sum >= remaining
    # Let's just iterate
    # To limit search space, cap the denominator
    max_d = start_d + 100000  # reasonable cap
    
    for d in range(start_d, min(start_d + 500, 10000000) + 1):
        if 1 / d > remaining:
            continue
        new_remaining = remaining - Fraction(1, d)
        if new_remaining < 0:
            break
        result = search(new_remaining, d + 1, terms_left - 1, chosen + [d])
        if result is not None:
            return result
    
    return None
