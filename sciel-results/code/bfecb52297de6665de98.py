"""Egyptian-fraction solver: find minimal-count representation of 1 with denominators >= 5."""
import time
from fractions import Fraction

def _try_dp(target, max_terms=None):
    """DP: find subset of divisors of target in [5, target] whose weights sum to target.
    Weight of d is target//d. Returns list of denominators or None."""
    if max_terms is None:
        max_terms = 256
    # BFS by number of terms to find minimal solution
    # State: (sum_of_weights, frozenset_of_denominators) - but that's too large
    # Instead: DP where reached[s] = min number of terms to reach sum s, and track one path
    from collections import defaultdict
    # reached[s] = (count, list_of_denominators)
    reached = {0: (0, [])}
    # Process denominators in order of decreasing weight (smaller d = larger weight)
    divisors = [d for d in range(5, target + 1) if target % d == 0]
    divisors.sort(key=lambda d: target // d, reverse=True)  # largest weight first
    
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

def _search_small_solution(deadline):
    """Try to find a solution with few terms using DP with various targets."""
    # Try targets with many divisors, starting with smaller ones
    # We want targets where the DP can find a short solution
    candidate_targets = [60, 120, 240, 360, 720, 840, 1260, 1680, 2520, 5040]
    
    best = None
    for t in candidate_targets:
        if time.time() > deadline:
            break
        result = _try_dp(t)
        if result is not None:
            # Verify
            total = sum(Fraction(1, d) for d in result)
            if total == 1 and all(5 <= d <= 10000000 for d in result) and len(set(result)) == len(result):
                if best is None or len(result) < len(best):
                    best = result
                if len(result) <= 5:  # Very short, likely minimal
                    break
    
    return best

def construct():
    """Return 1..256 distinct integer denominators in [5,10000000] whose reciprocal sum is exactly 1."""
    start = time.time()
    deadline = start + 80.0  # 80s safety margin within 90s limit
    
    # First, try to find a short solution using DP with various targets
    result = _search_small_solution(deadline)
    
    if result is not None:
        total = sum(Fraction(1, d) for d in result)
        if total == 1 and all(5 <= d <= 10000000 for d in result) and len(set(result)) == len(result) and 1 <= len(result) <= 256:
            return result
    
    # Fallback: use target=2520 DP (guaranteed to work)
    result = _try_dp(2520)
    if result is not None:
        total = sum(Fraction(1, d) for d in result)
        if total == 1 and all(5 <= d <= 10000000 for d in result) and len(set(result)) == len(result) and 1 <= len(result) <= 256:
            return result
    
    # Last resort: hardcoded known solution
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + 1/21 + 1/22 + 1/23 + 1/24 + 1/25 + 1/26 + 1/27 + 1/28 + 1/29 + 1/30 + 1/31 + 1/32 + 1/33 + 1/34 + 1/35 + 1/36 + 1/37 + 1/38 + 1/39 + 1/40
    # Let me verify this isn't 1... it won't be exactly 1.
    # Use the DP result as ultimate fallback
    raise RuntimeError("No valid Egyptian fraction found")
