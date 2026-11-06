# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def _greedy_complete(r, start_d, max_denom, used_set, max_terms=256):
    """Complete r using greedy, all denominators >= start_d, <= max_denom, distinct."""
    values = []
    r = Fraction(r)
    d_min = start_d
    while r > 0:
        if len(values) >= max_terms:
            return None
        if r.numerator == 0:
            break
        d = (r.denominator + r.numerator - 1) // r.numerator
        if d < d_min:
            d = d_min
        if d > max_denom:
            return None
        if d in used_set:
            d += 1
            while d <= max_denom and d in used_set:
                d += 1
            if d > max_denom:
                return None
        values.append(d)
        used_set.add(d)
        r -= Fraction(1, d)
        d_min = d + 1
    return values

def _dfs_complete(r, start_d, max_denom, used_set, depth, max_depth, max_terms, deadline):
    """Bounded DFS to complete r. Returns list of denominators or None.
    Only explores denominators <= max_denom, distinct, >= start_d."""
    if time.time() > deadline:
        return None
    if r == 0:
        return []
    if depth >= max_depth:
        # Fall back to greedy for the remainder
        return _greedy_complete(r, start_d, max_denom, used_set, max_terms)
    if r.numerator == 0:
        return []

    # 1/d <= r => d >= ceil(1/r) = ceil(r.denominator / r.numerator)
    d_lo = (r.denominator + r.numerator - 1) // r.numerator
    if d_lo < start_d:
        d_lo = start_d
    if d_lo > max_denom:
        return None

    # We also need enough remaining terms: if we have max_terms - len(used) - 1 left...
    # Just bound the search: only try a limited number of candidate d's
    # Try a window of candidate denominators starting from d_lo
    # To keep it bounded, try at most `window` values
    window = 50
    d_hi = min(d_lo + window, max_denom)

    for d in range(d_lo, d_hi + 1):
        if d in used_set:
            continue
        new_used = used_set | {d}
        new_r = r - Fraction(1, d)
        if new_r < 0:
            continue
        # remaining terms available
        remaining_terms = max_terms - len(used_set) - 1
        if remaining_terms <= 0:
            continue
        res = _dfs_complete(new_r, d + 1, max_denom, new_used,
                            depth + 1, max_depth, remaining_terms, deadline)
        if res is not None:
            return [d] + res
    return None

def _find_small_triplets(target, max_a, max_b, max_denom):
    """Find solutions of form 1/a + 1/b + 1/c = target with a < b < c <= max_denom."""
    best = None
    best_max = float('inf')
    for a in range(1, max_a + 1):
        ra = target - Fraction(1, a)
        if ra <= 0:
            continue
        for b in range(a + 1, max_b + 1):
            rb = ra - Fraction(1, b)
            if rb <= 0:
                continue
            # 1/c = rb => c = 1/rb
            if rb.numerator != 1:
                continue
            c = rb.denominator
            if c <= b:
                continue
            if c > max_denom:
                continue
            if c in (a, b):
                continue
            max_d = c
            if max_d < best_max:
                best_max = max_d
                best = [a, b, c]
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s

    target = Fraction(5, 31)
    best_solution = None
    best_max_denom = float('inf')

    # Phase 0: Explicit search for small triplets
    # 5/31 approx 0.16. a >= 7.
    # We search a up to 50, b up to 1000.
    triplet_sol = _find_small_triplets(target, 50, 1000, 10000000)
    if triplet_sol is not None:
        best_solution = triplet_sol
        best_max_denom = max(triplet_sol)

    # Phase 1: Greedy with various starting denominators
    min_d = 7
    max_d_upper = 10000000
    for first_d in range(min_d, min_d + 5000):
        if time.time() > deadline:
            break
        remainder = target - Fraction(1, first_d)
        if remainder < 0:
            break
        used = {first_d}
        completion = _greedy_complete(remainder, first_d + 1, max_d_upper, used)
        if completion is not None:
            all_denoms = [first_d] + completion
            max_denom = max(all_denoms)
            if max_denom < best_max_denom:
                best_max_denom = max_denom
                best_solution = all_denoms

    # Phase 2: DFS refinement
    if best_solution is not None:
        best_first = best_solution[0]
        search_range = range(max(min_d, best_first - 30), best_first + 30)
        for first_d in search_range:
            if time.time() > deadline:
                break
            remainder = target - Fraction(1, first_d)
            if remainder < 0:
                continue
            used = {first_d}
            max_allowed = best_max_denom - 1
            if max_allowed < first_d + 1:
                continue
            completion = _greedy_complete(remainder, first_d + 1, max_allowed, used)
            if completion is not None:
                all_denoms = [first_d] + completion
                max_denom = max(all_denoms)
                if max_denom < best_max_denom:
                    best_max_denom = max_denom
                    best_solution = all_denoms
            if time.time() < deadline:
                completion = _dfs_complete(remainder, first_d + 1, max_allowed,
                                           used, 0, 5, 256, deadline)
                if completion is not None:
                    all_denoms = [first_d] + completion
                    max_denom = max(all_denoms)
                    if max_denom < best_max_denom:
                        best_max_denom = max_denom
                        best_solution = all_denoms

    if best_solution is None:
        r = target
        values = []
        while r > 0:
            d = (r.denominator + r.numerator - 1) // r.numerator
            values.append(d)
            r -= Fraction(1, d)
        best_solution = values

    return best_solution
# EVOLVE-BLOCK-END
