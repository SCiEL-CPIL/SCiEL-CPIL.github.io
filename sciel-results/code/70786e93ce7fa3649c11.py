"""Rewritten: minimize the largest denominator in an Egyptian expansion of 5/31."""
import time

# EVOLVE-BLOCK-START
from fractions import Fraction
from math import ceil, floor

_TARGET = Fraction(5, 31)
_DEADLINE_MARGIN = 5.0  # seconds to keep before the 90s wall-clock kill

def _feasible(r, d_min, D, max_terms, deadline):
    """Return a list of denominators (the chosen path) if r can be written as a sum
    of at most max_terms distinct unit fractions with each in [d_min, D], else None.
    Uses exact Fraction arithmetic and strong pruning."""
    if r == 0:
        return []
    if max_terms <= 0:
        return None
    if time.time() > deadline:
        return None

    # Prune: with up to max_terms terms each in [d_min, D],
    # the achievable sum is in [max_terms/D, max_terms/d_min].
    if r > Fraction(max_terms, d_min):
        return None
    if r < Fraction(max_terms, D):
        return None

    # The next denominator d must satisfy:
    #   d >= d_min
    #   d <= D
    #   1/d <= r  (so r - 1/d >= 0)  =>  d >= ceil(1/r)
    #   after choosing d, the remainder r' = r - 1/d must be representable
    #     by at most max_terms-1 terms in [d+1, D].
    # Lower bound on d from 1/d <= r:  d >= ceil(1/r)
    lo = max(d_min, (r.denominator + r.numerator - 1) // r.numerator)
    if lo > D:
        return None
    hi = D
    # Also, to leave a remainder that can be covered by max_terms-1 terms each <= D:
    #   r - 1/d >= (max_terms-1)/D  =>  1/d <= r - (max_terms-1)/D
    #   => d >= ceil(1 / (r - (max_terms-1)/D))  if r > (max_terms-1)/D
    rem_min = Fraction(max_terms - 1, D)
    if r > rem_min:
        gap = r - rem_min
        if gap > 0:
            lo2 = (gap.denominator + gap.numerator - 1) // gap.numerator
            if lo2 > lo:
                lo = lo2
    if lo > hi:
        return None

    # Upper bound on d from the fact that the remainder must be >= (max_terms-1)/D
    # is already handled by lo. Also remainder must be >= 0 handled by lo.
    # Additionally, the remainder r' must be <= (max_terms-1)/(d+1) to be coverable
    # by terms each >= d+1.  =>  r - 1/d <= (max_terms-1)/(d+1)
    # This gives an upper bound on d. Solve:
    #   r - 1/d <= (max_terms-1)/(d+1)
    # Multiply by d(d+1) > 0:
    #   r*d*(d+1) - (d+1) <= (max_terms-1)*d
    #   r*d^2 + r*d - d - 1 <= (max_terms-1)*d
    #   r*d^2 + (r - 1 - (max_terms-1))*d - 1 <= 0
    #   r*d^2 + (r - max_terms)*d - 1 <= 0
    # This is a quadratic in d; find the positive root for the upper bound.
    # r*d^2 + (r - max_terms)*d - 1 <= 0
    # Use exact: find largest integer d with r*d^2 + (r-max_terms)*d - 1 <= 0.
    # Since r>0, the quadratic opens upward; the valid region is between the two
    # roots. The positive root is the upper bound.
    # root = [-(r-max_terms) + sqrt((r-max_terms)^2 + 4r)] / (2r)
    # We'll just cap hi by D and rely on the per-branch prune; but a tighter hi
    # helps. Compute an approximate upper bound via floats for speed, then verify.
    rr = float(r)
    mm = float(max_terms)
    disc = (rr - mm) ** 2 + 4.0 * rr
    if disc >= 0:
        root = (-(rr - mm) + disc ** 0.5) / (2.0 * rr)
        hi2 = int(root) + 2
        if hi2 < hi:
            hi = hi2
    if lo > hi:
        return None

    # Try denominators. Searching from a good heuristic start helps.
    # Greedy start: d0 = ceil(1/r)
    d0 = (r.denominator + r.numerator - 1) // r.numerator
    if d0 < lo:
        d0 = lo
    if d0 > hi:
        d0 = hi
    # Build candidate order: start near d0, then expand outward.
    cands = []
    cands.append(d0)
    for off in range(1, hi - lo + 1):
        a = d0 - off
        b = d0 + off
        if a >= lo:
            cands.append(a)
        if b <= hi:
            cands.append(b)
        if a < lo and b > hi:
            break

    for d in cands:
        if d < lo or d > hi:
            continue
        r2 = r - Fraction(1, d)
        if r2 < 0:
            continue
        # Quick prune on r2 with remaining terms
        if max_terms - 1 > 0:
            if r2 > Fraction(max_terms - 1, d + 1):
                continue
            if r2 < Fraction(max_terms - 1, D):
                continue
        sub = _feasible(r2, d + 1, D, max_terms - 1, deadline)
        if sub is not None:
            return [d] + sub
    return None

def _solve_for_D(D, max_terms, deadline):
    """Try to find an expansion of _TARGET with all denominators in [1, D],
    at most max_terms terms. Returns list of denominators or None."""
    return _feasible(_TARGET, 1, D, max_terms, deadline)

def construct():
    t0 = time.time()
    deadline = t0 + 90.0 - _DEADLINE_MARGIN

    # Lower bound on the largest denominator: at least ceil(1 / (5/31)) = ceil(31/5) = 7.
    # But we need distinct denominators summing to 5/31 > 1/2, so we need several terms.
    # Start D from a small value and increase until feasible.
    # Use a generous max_terms (<=256). Start with a moderate max_terms and grow.
    # To minimize D, we search D increasing; for each D try to find ANY valid expansion.
    # Use max_terms up to 256 but the DFS depth is bounded by max_terms; start with
    # a smaller effective max_terms to keep the search tractable, then increase.

    best = None
    best_D = None

    # Try increasing D. For each D, try increasing max_terms until found or budget.
    D = 7
    max_D = 10000000
    while D <= max_D and time.time() < deadline:
        # Determine a sensible max_terms: more terms make it easier but deeper.
        # Start with a small max_terms and increase.
        for max_terms in (4, 5, 6, 8, 10, 14, 20, 30, 50, 100, 256):
            if time.time() > deadline:
                break
            res = _solve_for_D(D, max_terms, deadline)
            if res is not None:
                # Verify
                s = sum(Fraction(1, x) for x in res)
                if s == _TARGET and len(set(res)) == len(res) and all(1 <= x <= D for x in res):
                    if best is None or D < best_D:
                        best = res
                        best_D = D
                    return res
        D += 1

    if best is not None:
        return best
    # Fallback: greedy (seed behavior)
    r = _TARGET
    values = []
    while r:
        d = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d)
        r -= Fraction(1, d)
    return values
# EVOLVE-BLOCK-END
