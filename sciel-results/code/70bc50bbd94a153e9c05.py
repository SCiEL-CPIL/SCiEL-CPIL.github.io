"""Egyptian-fraction solver: find minimal-count representation of 1 with denominators >= 5."""
import time
from fractions import Fraction

def _recursive_search(remaining, min_denom, max_terms, used, deadline):
    """
    Recursively search for a set of distinct denominators >= min_denom
    whose reciprocals sum to 'remaining', using at most max_terms terms.
    Returns list of denominators or None.
    """
    if time.time() > deadline:
        return None
    if max_terms == 0:
        return [] if remaining == 0 else None
    if remaining == 0:
        return []
    if remaining < 0:
        return None
    
    # The next denominator must be at least min_denom
    # Also, 1/d <= remaining => d >= ceil(1/remaining)
    # And we need max_terms * (1/min_denom) >= remaining for a solution to exist
    # i.e., min_denom <= max_terms / remaining
    
    # Upper bound on denominator: 1/d <= remaining => d >= 1/remaining
    # Lower bound: d >= min_denom
    # Also, to have enough remaining terms cover the rest:
    # (max_terms - 1) * (1/(d+1)) >= remaining - 1/d  (rough bound)
    
    import math
    lower_d = max(min_denom, math.ceil(1 / remaining))
    # Upper bound: we need 1/d + (max_terms-1)/(d+1) >= remaining
    # For a loose upper bound: d <= max_terms / remaining
    upper_d = int(max_terms / remaining) + 1
    upper_d = min(upper_d, 10000000)
    
    for d in range(lower_d, upper_d + 1):
        if d in used:
            continue
        new_remaining = remaining - Fraction(1, d)
        if new_remaining < 0:
            break
        result = _recursive_search(new_remaining, d + 1, max_terms - 1, used | {d}, deadline)
        if result is not None:
            return [d] + result
    return None

def _find_short_solution(deadline):
    """Try to find a short solution using recursive search."""
    # Try increasing max_terms
    for max_terms in range(1, 20):
        if time.time() > deadline:
            break
        result = _recursive_search(Fraction(1, 1), 5, max_terms, set(), deadline)
        if result is not None:
            # Verify
            total = sum(Fraction(1, d) for d in result)
            if total == 1 and all(5 <= d <= 10000000 for d in result) and len(set(result)) == len(result):
                return result
    return None

def construct():
    """Return 1..256 distinct integer denominators in [5,10000000] whose reciprocal sum is exactly 1."""
    start = time.time()
    deadline = start + 85.0  # 85s safety margin within 90s limit
    
    # Try to find a short solution using recursive search
    result = _find_short_solution(deadline)
    
    if result is not None:
        total = sum(Fraction(1, d) for d in result)
        if total == 1 and all(5 <= d <= 10000000 for d in result) and len(set(result)) == len(result) and 1 <= len(result) <= 256:
            return result
    
    # Fallback: use a known valid solution
    # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + 1/21 + 1/22 + 1/23 + 1/24 + 1/25 + 1/26 + 1/27 + 1/28 + 1/29 + 1/30 + 1/31 + 1/32 + 1/33 + 1/34 + 1/35 + 1/36 + 1/37 + 1/38 + 1/39 + 1/40 + 1/41 + 1/42 + 1/43 + 1/44 + 1/45 + 1/46 + 1/47 + 1/48 + 1/49 + 1/50 + 1/51 + 1/52 + 1/53 + 1/54 + 1/55 + 1/56 + 1/57 + 1/58 + 1/59 + 1/60 + 1/61 + 1/62 + 1/63 + 1/64 + 1/65 + 1/66 + 1/67 + 1/68 + 1/69 + 1/70 + 1/71 + 1/72 + 1/73 + 1/74 + 1/75 + 1/76 + 1/77 + 1/78 + 1/79 + 1/80 + 1/81 + 1/82 + 1/83 + 1/84 + 1/85 + 1/86 + 1/87 + 1/88 + 1/89 + 1/90 + 1/91 + 1/92 + 1/93 + 1/94 + 1/95 + 1/96 + 1/97 + 1/98 + 1/99 + 1/100
    # This won't sum to exactly 1. Let me use a verified solution.
    
    # Use the DP approach as fallback
    def _try_dp(target, max_terms=256):
        reached = {0: (0, [])}
        divisors = [d for d in range(5, target + 1) if target % d == 0]
        divisors.sort(key=lambda d: target // d, reverse=True)
        for d in divisors:
            w = target // d
            new_reached = dict(reached)
            for s, (cnt, dens) in reached.items():
                ns = s + w
                if ns <= target:
                    nc = cnt + 1
                    if nc <= max_terms:
                        if ns not in new_reached or nc < new_reached[ns][0]:
                            new_reached[ns] = (nc, dens + [d])
            reached = new_reached
            if target in reached:
                return reached[target][1]
        return None
    
    result = _try_dp(2520)
    if result is not None:
        total = sum(Fraction(1, d) for d in result)
        if total == 1 and all(5 <= d <= 10000000 for d in result) and len(set(result)) == len(result) and 1 <= len(result) <= 256:
            return result
    
    raise RuntimeError("No valid Egyptian fraction found")
