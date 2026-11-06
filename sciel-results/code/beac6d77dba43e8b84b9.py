import time
from math import gcd

def construct():
    deadline = time.time() + 80.0  # safety margin before 90s limit

    # Known minimal solutions for 1/n = 1/a + 1/b + ... with a,b,c,d > 4
    # Try known short representations first (hardcoded verified solutions)
    # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/13 + 1/15 + 1/20 + 1/30 + 1/42 + 1/65 + 1/110 + 1/195
    # Actually let's search systematically for small count solutions.

    # Strategy: BFS/DFS for representations of 1 as sum of unit fractions with denominators > 4
    # Minimize the number of terms.

    # Start with known small solutions and verify, then search for better ones.

    # A well-known solution: 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/13 + 1/15 + 1/20 + 1/30 + 1/42 + 1/65 + 1/110 + 1/195
    # That's 15 terms. Can we do better?

    # Let's use a systematic search with exact rational arithmetic (fractions).
    # We'll search for representations with k terms where k is minimized.

    from fractions import Fraction

    def search(target, min_denom, max_terms, current, best):
        """DFS to find representation of target as sum of unit fractions."""
        if time.time() > deadline - 2.0:
            return
        if not current:
            if best is not None:
                return best
            return None
        if len(current) >= max_terms:
            if current and target == 0:
                return current
            return None

        # Try next denominator starting from min_denom
        # Upper bound: target = 1/d, so d >= ceil(1/target) if target > 0
        # Also d must be >= min_denom
        if target <= 0:
            return None

        # d must satisfy 1/d <= target, i.e., d >= 1/target
        # Also to make progress, 1/d < target (strictly, since we need more terms or this is the last)
        # If this is the last term, 1/d == target, so d = 1/target
        if len(current) == max_terms - 1:
            # Last term must equal target exactly
            if target.denominator == 1 or Fraction(1).denominator == 1:
                pass
            d_candidate = target.denominator if target.numerator == 1 else None
            # target = n/d, we need 1/x = n/d => x = d/n, only works if n=1
            if target.numerator == 1:
                d = target.denominator
                if d >= min_denom and d >= 5 and d <= 10000000 and d not in set(current):
                    return current + [d]
            return None

        # For non-last terms: 1/d < target (since remaining terms are positive)
        # So d > 1/target
        # Also 1/d <= target / (remaining_terms) for feasibility... actually:
        # We need target - 1/d to be representable with (max_terms - len(current) - 1) terms
        # Each term <= 1/min_denom_next, so target - 1/d <= (remaining-1) / min_denom_next
        # This gives lower bound on d.

        # Upper bound on d: 1/d < target => d > 1/target
        # Lower bound: we want d >= min_denom
        # Also for feasibility with remaining terms: target - 1/d must be achievable
        # with (max_terms - len(current) - 1) terms each <= 1/min_denom
        # So target - 1/d <= (max_terms - len(current) - 1) / min_denom
        # => 1/d >= target - (max_terms - len(current) - 1) / min_denom
        # => d <= 1 / (target - (max_terms - len(current) - 1) / min_denom) if denominator positive

        remaining = max_terms - len(current) - 1
        if remaining < 0:
            return None

        # d must be at least min_denom
        d_start = max(min_denom, 5)

        # d must be > 1/target (i.e., 1/d < target) for non-last terms
        # 1/d < target => d > 1/target => d > target.denominator / target.numerator
        # Since target = n/d_g, 1/target = d_g/n
        # So d > d_g/n, meaning d >= floor(d_g/n) + 1

        d_min_from_target = target.denominator // target.numerator + 1
        d_start = max(d_start, d_min_from_target)

        # Upper bound: target - 1/d <= remaining / min_denom (each remaining term <= 1/min_denom)
        # 1/d >= target - remaining / min_denom
        # If target - remaining/min_denom <= 0, no upper bound from this
        if target > Fraction(remaining, min_denom):
            diff = target - Fraction(remaining, min_denom)
            if diff > 0:
                # 1/d >= diff => d <= 1/diff = 1/(n/d) = d/n
                d_max = diff.denominator // diff.numerator
                # But also d must be such that 1/d < target, already handled by d_start
                if d_max < d_start:
                    return None
            else:
                d_max = 10000000
        else:
            d_max = 10000000

        # Also cap d_max at 10000000
        d_max = min(d_max, 10000000)

        current_set = set(current)

        for d in range(d_start, d_max + 1):
            if d in current_set:
                continue
            if time.time() > deadline - 2.0:
                return best if best is not None else None

            new_target = target - Fraction(1, d)
            if new_target < 0:
                break
            if new_target == 0:
                # Found a solution!
                result = current + [d]
                if best is None or len(result) < len(best):
                    best = result
                return best

            # Recursive search with minimum denominator > d (to ensure distinct and ordered)
            result = search(new_target, d + 1, max_terms, current + [d], best)
            if result is not None:
                best = result
                # If we found a solution, try to find a shorter one
                # But for now, continue searching for potentially shorter ones
                # Actually, since we're going by increasing d, we might find different solutions
                # Let's continue to see if there's a shorter one (fewer total terms)
                # But max_terms is fixed, so all solutions here have the same length
                # We should stop if we've used all terms
                if len(result) == max_terms:
                    # Can't do better with this max_terms
                    # But maybe with fewer max_terms we could? That's handled by outer loop
                    break

        return best

    # Try increasing number of terms starting from small
    # Minimum possible: we need sum of reciprocals > 1 with denominators >= 5
    # 1/5 + 1/5 + 1/5 + 1/5 = 4/5 < 1, so at least 3 terms won't work easily
    # Actually 1/5+1/5+1/5 = 3/5, 1/5+1/6+1/7 ≈ 0.2+0.167+0.143 = 0.51
    # We need enough terms. Let's start from k=5 and go up.

    best_result = None
    for k in range(5, 20):
        if time.time() > deadline - 5.0:
            break
        result = search(Fraction(1, 1), 5, k, [], None)
        if result is not None:
            best_result = result
            break  # Found minimal k

    if best_result is None:
        # Fallback: use a known valid solution
        # Verify: 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/13 + 1/15 + 1/20 + 1/30 + 1/42 + 1/65 + 1/110 + 1/195
        fallback = [5, 6, 7, 8, 9, 10, 11, 13, 15, 20, 30, 42, 65, 110, 195]
        s = sum(Fraction(1, d) for d in fallback)
        if s == 1:
            best_result = fallback
        else:
            # Try another known solution
            # 1 = 1/2 + 1/3 + 1/6 but 2,3 < 5
            # 1 = 1/5 + 1/5 + 1/5 + 1/5 + 1/5 not distinct
            # Let's compute a valid one
            # 1 = 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + ... 
            # Better to just do a quick DP or use the original approach as fallback
            target = 2520
            reached = {0: []}
            for d in range(5, target + 1):
                if target % d:
                    continue
                w = target // d
                for total, values in list(reached.items()):
                    if total + w <= target and total + w not in reached:
                        reached[total + w] = values + [d]
                if target in reached:
                    best_result = reached[target]
                    break

    if best_result is None:
        raise RuntimeError("no representation found")

    # Verify
    s = sum(Fraction(1, d) for d in best_result)
    assert s == 1, f"Sum is {s}, not 1"
    assert len(best_result) >= 1 and len(best_result) <= 256
    assert all(5 <= d <= 10000000 for d in best_result)
    assert len(set(best_result)) == len(best_result)

    return best_result
