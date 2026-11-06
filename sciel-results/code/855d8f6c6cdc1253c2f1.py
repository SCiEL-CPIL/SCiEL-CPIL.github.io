"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import gcd, ceil

# Precompute prefix sums of reciprocals for fast max-sum queries
# H[n] = sum_{i=0}^{n-1} 1/(5+i) for n up to 256
_PREFIX_SUM = [Fraction(0, 1)]
for _i in range(256):
    _PREFIX_SUM.append(_PREFIX_SUM[-1] + Fraction(1, 5 + _i))

def _compute_max_sum(k, min_den):
    """Compute maximum possible sum of k distinct denominators starting from min_den."""
    if k <= 0:
        return Fraction(0, 1)
    # Sum of 1/min_den + 1/(min_den+1) + ... + 1/(min_den+k-1)
    # = H[min_den+k-1] - H[min_den-1] where H[n] = sum_{i=1}^{n} 1/i
    # But we precomputed from 5, so let's use a different approach
    # Actually, let's just compute it directly but cache results
    key = (k, min_den)
    if key in _max_sum_cache:
        return _max_sum_cache[key]
    total = Fraction(0, 1)
    for i in range(k):
        total += Fraction(1, min_den + i)
    _max_sum_cache[key] = total
    return total

_max_sum_cache = {}

def _compute_min_sum(k, min_den, max_den_limit=10000000):
    """Compute minimum possible sum of k distinct denominators >= min_den, bounded by max_den_limit."""
    if k <= 0:
        return Fraction(0, 1)
    # Use the k largest possible denominators >= min_den
    # They are: max_den_limit, max_den_limit-1, ..., max_den_limit-k+1
    # But if max_den_limit - k + 1 < min_den, we need to adjust
    total = Fraction(0, 1)
    for i in range(k):
        d = max_den_limit - i
        if d < min_den:
            # This shouldn't happen if k is small enough
            d = min_den + (k - 1 - i)
        total += Fraction(1, d)
    return total

def _search_min_terms():
    """Find minimum number of distinct denominators >= 5 summing to 1."""
    deadline = time.time() + 85  # Safety margin before 90s kill
    best = None
    _max_sum_cache.clear()

    def dfs(remaining, min_den, current, terms_left):
        nonlocal best
        if time.time() > deadline:
            return
        if terms_left == 0:
            if remaining == 0:
                if best is None or len(current) < len(best):
                    best = list(current)
            return
        if remaining <= 0:
            if remaining == 0 and terms_left == 0:
                if best is None or len(current) < len(best):
                    best = list(current)
            return

        # Pruning: check if remaining is achievable with terms_left terms
        # Max possible sum: use terms_left smallest available denominators
        max_possible = _compute_max_sum(terms_left, min_den)
        if remaining > max_possible:
            return

        # Min possible sum check: use terms_left largest possible denominators
        # The k largest denominators >= min_den are: 10^7, 10^7-1, ..., 10^7-k+1
        # Their sum is very small, so this bound is usually not tight
        # But for small terms_left, it might help
        if terms_left <= 2:
            min_possible = _compute_min_sum(terms_left, min_den)
            if remaining < min_possible:
                return

        if terms_left == 1:
            # Directly compute the required denominator
            if remaining > 0:
                # 1/d = remaining => d = 1/remaining
                # remaining must be a unit fraction
                num, den = remaining.numerator, remaining.denominator
                if num == 1:
                    d = den
                    if d >= min_den and d <= 10000000 and d not in current:
                        if best is None or len(current) + 1 < len(best):
                            best = list(current) + [d]
            return

        # Try denominators for the current term
        # d must satisfy: 1/d <= remaining => d >= ceil(1/remaining)
        # Also d >= min_den
        if remaining > 0:
            num, den = remaining.numerator, remaining.denominator
            min_d = max(min_den, den // num + (1 if den % num != 0 else 0))
        else:
            min_d = min_den

        # Upper bound on d: we need remaining - 1/d to be achievable with (terms_left-1) terms
        # The max sum of (terms_left-1) terms with min denominator d+1 must be >= remaining - 1/d
        # i.e., remaining - 1/d <= _compute_max_sum(terms_left-1, d+1)
        # Also, 1/d <= remaining => d >= 1/remaining (already handled)
        # And d cannot exceed 10^7
        
        # Tighter upper bound: 
        # We need remaining - 1/d >= min possible sum of (terms_left-1) terms >= min_den'
        # But more usefully: 1/d must be at least remaining/terms_left (rough lower bound on each term)
        # So d <= terms_left / remaining
        max_d = min(10000000, int(terms_left / remaining) + 1) if remaining > 0 else 10000000

        # Additionally, we can tighten max_d using the max-sum constraint:
        # remaining - 1/d <= _compute_max_sum(terms_left-1, d+1)
        # For large d, 1/d is small, so remaining - 1/d ≈ remaining
        # And _compute_max_sum(terms_left-1, d+1) decreases as d increases
        # So there's a cutoff beyond which no solution exists
        # Let's find it by binary search or just use the loose bound
        
        # Iterate through possible d values
        # Heuristic: try d values in order of likelihood
        # Prioritize d where (remaining - 1/d) has small numerator (easier to decompose)
        
        for d in range(min_d, max_d + 1):
            if time.time() > deadline:
                return
            new_rem = remaining - Fraction(1, d)
            if new_rem < 0:
                break
            if new_rem == 0:
                if terms_left == 1:
                    if best is None or len(current) + 1 < len(best):
                        best = list(current) + [d]
                continue

            # Check if new_rem is achievable with (terms_left - 1) terms
            if terms_left - 1 > 0:
                max_possible_next = _compute_max_sum(terms_left - 1, d + 1)
                if new_rem > max_possible_next:
                    continue
                # Check minimum sum for tighter pruning
                if terms_left - 1 <= 2:
                    min_possible_next = _compute_min_sum(terms_left - 1, d + 1)
                    if new_rem < min_possible_next:
                        continue

            dfs(new_rem, d + 1, current + [d], terms_left - 1)

    # Determine the minimum number of terms needed
    # 7 terms: max sum with 5,6,7,8,9,10,11 = ~0.937 < 1, so 7 is impossible
    # 8 terms: max sum with 5,6,7,8,9,10,11,12 = ~1.02 > 1, so 8 might work
    for n_terms in range(8, 13):
        # Check if n_terms is even possible
        max_sum = _compute_max_sum(n_terms, 5)
        if max_sum < 1:
            continue
        dfs(Fraction(1, 1), 5, [], n_terms)
        if best is not None:
            break
        if time.time() > deadline:
            break

    return best

def construct():
    result = _search_min_terms()
    if result is None:
        # Fallback: use a known solution or DP approach
        # Try a simple DP with a moderate common denominator
        target = 2520
        reached = {0: []}
        for d in range(5, target + 1):
            if target % d != 0:
                continue
            w = target // d
            for total, values in list(reached.items()):
                new_total = total + w
                if new_total <= target and new_total not in reached:
                    reached[new_total] = values + [d]
            if target in reached:
                return reached[target]
        # If still no solution, return a hardcoded valid solution
        raise RuntimeError("no baseline representation")
    return result
# EVOLVE-BLOCK-END
