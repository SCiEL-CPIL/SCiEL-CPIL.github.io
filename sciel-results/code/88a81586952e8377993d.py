import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    target = Fraction(5, 31)

    # Known valid solution: 1/7 + 1/56 + 1/1736 = 5/31
    best_solution = [7, 56, 1736]
    best_max = 1736

    def verify(denoms):
        if not denoms or len(denoms) > 256:
            return False
        if len(set(denoms)) != len(denoms):
            return False
        if any(d < 1 or d > 10000000 for d in denoms):
            return False
        s = sum(Fraction(1, d) for d in denoms)
        return s == target

    def greedy_complete(remaining, min_d, max_allowed, current, max_terms=8):
        """Greedy: always pick smallest valid d, then DFS to fix remainder."""
        if time.time() > deadline:
            return None
        if remaining == 0:
            if current and verify(current):
                return list(current)
            return None
        if len(current) >= max_terms:
            return None

        # Lower bound on d
        d_lo = (remaining.denominator + remaining.numerator - 1) // remaining.numerator
        if d_lo < min_d:
            d_lo = min_d
        if d_lo > max_allowed:
            return None

        # Try greedy: pick d_lo first
        new_rem = remaining - Fraction(1, d_lo)
        if new_rem >= 0:
            # Check if new_rem can be expressed with (max_terms - len(current) - 1) terms
            k_left = max_terms - len(current) - 1
            if new_rem == 0:
                result = list(current) + [d_lo]
                if verify(result):
                    return result
            elif k_left > 0:
                # Bounded DFS on the remainder
                result = _dfs(new_rem, d_lo + 1, max_allowed, list(current) + [d_lo], k_left, max_terms)
                if result:
                    return result

        # If greedy d_lo didn't work, try a few more values
        for d in range(d_lo + 1, min(d_lo + 200, max_allowed + 1)):
            if time.time() > deadline:
                return None
            new_rem = remaining - Fraction(1, d)
            if new_rem < 0:
                break
            k_left = max_terms - len(current) - 1
            if new_rem == 0:
                result = list(current) + [d]
                if verify(result):
                    return result
            elif k_left > 0:
                result = _dfs(new_rem, d + 1, max_allowed, list(current) + [d], k_left, max_terms)
                if result:
                    return result
        return None

    def _dfs(remaining, min_d, max_allowed, current, k_left, max_total_terms):
        """Bounded DFS for completing the expansion."""
        if time.time() > deadline:
            return None
        if remaining == 0:
            if current and verify(current):
                return list(current)
            return None
        if k_left <= 0:
            return None
        if len(current) >= max_total_terms:
            return None

        d_lo = (remaining.denominator + remaining.numerator - 1) // remaining.numerator
        if d_lo < min_d:
            d_lo = min_d
        if d_lo > max_allowed:
            return None

        # If only one term left, must be exact
        if k_left == 1:
            if remaining.numerator == 1:
                d_exact = remaining.denominator
                if d_exact >= d_lo and d_exact <= max_allowed:
                    result = list(current) + [d_exact]
                    if verify(result):
                        return result
            return None

        # Prune: remaining must be <= k_left / min_d (each term <= 1/min_d)
        # Actually each term <= 1/(d_lo), so remaining <= k_left / d_lo
        # And remaining >= k_left / max_allowed (each term >= 1/max_allowed)
        if remaining > Fraction(k_left, d_lo):
            return None
        if remaining < Fraction(k_left, max_allowed):
            return None

        # Upper bound on d: 1/d >= remaining - (k_left-1)/max_allowed
        # => d <= 1/(remaining - (k_left-1)/max_allowed) if positive
        # Simpler: iterate d from d_lo, limit range
        d_hi = min(max_allowed, d_lo + 300)

        for d in range(d_lo, d_hi + 1):
            if time.time() > deadline:
                return None
            new_rem = remaining - Fraction(1, d)
            if new_rem < 0:
                break
            if new_rem == 0:
                if k_left == 1:
                    result = list(current) + [d]
                    if verify(result):
                        return result
                # If k_left > 1 and new_rem == 0, we'd need 0 terms for 0, invalid unless we allow it
                # Actually if new_rem == 0 and we still have k_left-1 > 0 terms to place, that's bad
                # unless we can place them as... no, they must sum to 0 which is impossible with positive terms
                # So only valid if k_left-1 == 0
                continue
            # Prune: new_rem must be expressible with k_left-1 terms in [d+1, max_allowed]
            if k_left - 1 > 0:
                if new_rem > Fraction(k_left - 1, d + 1):
                    continue
                if new_rem < Fraction(k_left - 1, max_allowed):
                    continue
            result = _dfs(new_rem, d + 1, max_allowed, list(current) + [d], k_left - 1, max_total_terms)
            if result:
                return result
        return None

    def find_solution_with_max(K, max_terms=8):
        """Find any valid expansion of 5/31 with all denominators <= K."""
        if time.time() > deadline:
            return None
        return greedy_complete(target, 1, K, [], max_terms)

    # Binary search on K: find minimum K such that a solution exists
    # We know K=1736 works. Try to find smaller K.
    # Lower bound: need at least 1 term, 1/d <= 5/31 => d >= 7. So K >= 7.
    # But with multiple terms, K must be at least ceil(31/5) = 7 for single term.
    # For 2 terms: 1/a + 1/b = 5/31. Try to see if 2-term exists.
    # For 3 terms: known 7,56,1736.
    
    # First, check if a 2-term solution exists
    # 1/a + 1/b = 5/31 => (a+b)/(ab) = 5/31 => 31(a+b) = 5ab
    # => 5ab - 31a - 31b = 0 => (5a-31)(5b-31) = 31*31 = 961
    # 961 = 1*961, 31*31
    # Case 1: 5a-31=1, 5b-31=961 => a=32/5 not integer
    # Case 2: 5a-31=31, 5b-31=31 => a=62/5 not integer
    # Case 3: 5a-31=961, 5b-31=1 => same as case 1
    # So no 2-term solution.

    # Try 3-term solutions with smaller max
    # 1/a + 1/b + 1/c = 5/31
    # We'll use binary search on K for 3-term solutions first (faster), then 4+, then more
    
    # Binary search for minimum K with 3 terms
    lo, hi = 7, 1735  # hi is best_max - 1, we want to find if we can beat 1736
    best_3term = None
    
    # Check if 3-term exists with max <= hi
    # For 3 terms, a < b < c <= K
    # a >= 7, and 1/a <= 5/31 => a >= 7
    # Also 3/a >= 5/31 => a <= 93/5 = 18.6, so a <= 18
    # For each a in [7,18], 1/b + 1/c = 5/31 - 1/a
    # This is a 2-term problem for the remainder
    
    for a in range(7, 19):
        if time.time() > deadline:
            break
        ra = target - Fraction(1, a)
        if ra <= 0:
            continue
        # 1/b + 1/c = ra, b > a, c > b, c <= K
        # (b+c)/(bc) = ra => ra*bc = b+c => (ra*b - 1)(ra*c - 1) = 1... 
        # Better: ra = p/q in lowest terms. Then p*bc = q*b + q*c
        # => p*bc - q*b - q*c = 0 => (p*b - q)(p*c - q) = q^2
        p, q = ra.numerator, ra.denominator
        # Factor q^2
        q2 = q * q
        # Find factor pairs (d1, d2) with d1*d2 = q2, d1 <= d2
        # b = (d1 + q) / p, c = (d2 + q) / p
        # Must have b > a, c > b, c <= K
        found = False
        for d1 in range(1, int(q2**0.5) + 1):
            if q2 % d1 != 0:
                continue
            d2 = q2 // d1
            # b = (d1 + q) / p must be integer
            if (d1 + q) % p != 0:
                continue
            b = (d1 + q) // p
            if (d2 + q) % p != 0:
                continue
            c = (d2 + q) // p
            if b <= a or c <= b:
                continue
            if b < 1 or c > 10000000:
                continue
            if c <= 1735:  # We're looking for max < 1736
                sol = [a, b, c]
                if verify(sol):
                    if best_3term is None or max(sol) < max(best_3term):
                        best_3term = sol
                    found = True
        if found:
            # Found a 3-term with max < 1736
            pass

    if best_3term:
        best_solution = best_3term
        best_max = max(best_solution)

    # Now binary search for even smaller K with 4+ terms
    # Use binary search: can we find a solution with max <= K?
    # Try K values: start with best_max - 1, go down
    
    # For efficiency, try a few specific K values with increasing term counts
    current_best_K = best_max
    for K_test in range(current_best_K - 1, max(100, current_best_K - 500), -1):
        if time.time() > deadline - 5:
            break
        # Try with up to 8 terms
        sol = find_solution_with_max(K_test, max_terms=8)
        if sol:
            best_solution = sol
            best_max = K_test
            current_best_K = K_test
            # Continue searching for even smaller K
            break
    
    # If we found a better solution, do binary search between 100 and best_max
    if best_max < 1736:
        lo2, hi2 = 100, best_max - 1
        while lo2 <= hi2 and time.time() < deadline - 3:
            mid = (lo2 + hi2) // 2
            sol = find_solution_with_max(mid, max_terms=8)
            if sol:
                best_solution = sol
                best_max = mid
                hi2 = mid - 1
            else:
                lo2 = mid + 1

    # Final verification
    if not verify(best_solution):
        best_solution = [7, 56, 1736]
        best_max = 1736

    return best_solution
