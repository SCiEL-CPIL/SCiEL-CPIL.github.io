"""Bootstrap portfolio: MODEL-AND-OPTIMIZE for Egyptian fraction of 5/31."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
from math import gcd
import sys

TARGET = Fraction(5, 31)
MAX_DENOM = 10_000_000
MAX_TERMS = 256
DEADLINE = 55.0  # Safety margin for 60s limit

def _try_solution(denoms):
    """
    Validate a list of denominators and return the largest one if valid, else None.
    """
    if not denoms:
        return None
    if len(denoms) > MAX_TERMS:
        return None
    if len(set(denoms)) != len(denoms):
        return None
    if any(d < 1 or d > MAX_DENOM for d in denoms):
        return None
    
    total = sum(Fraction(1, d) for d in denoms)
    if total != TARGET:
        return None
    
    return max(denoms)

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Known high-quality solution: [8, 31, 248] -> max 248
    known_sol = [8, 31, 248]
    best_max = _try_solution(known_sol)
    best_list = known_sol
    
    if best_max is None:
        return known_sol

    # Fast bounded search for better solutions
    def _search_bounded(rem, min_d, depth, limit_max, used):
        """
        Find Egyptian fraction of 'rem' using denominators >= min_d, all < limit_max,
        not in 'used', with at most MAX_TERMS - depth more terms.
        Returns list of denominators or None.
        """
        if time.time() > deadline:
            return None
        if rem == 0:
            return []
        if depth >= MAX_TERMS:
            return None
        
        # Lower bound: need 1/d <= rem for positive remainder (for first term)
        # Actually we can have 1/d > rem if we use multiple terms, but for efficiency
        # let's use the standard bound: d >= ceil(1/rem)
        min_d_eff = max(min_d, (rem.denominator + rem.numerator - 1) // rem.numerator)
        if min_d_eff >= limit_max:
            return None
        
        # Upper bound on d: we need enough remaining terms to sum to rem - 1/d
        # With k terms left (after picking d), max sum is 1/(d+1) + 1/(d+2) + ... 
        # But simpler: if rem - 1/d > 0, we need at least 1 more term.
        # A practical upper bound: d cannot be too large or rem - 1/d becomes too small
        # to be represented with remaining terms under limit_max.
        
        # Also: if we pick d, then rem_new = rem - 1/d.
        # If rem_new > 0, we need to represent it with (MAX_TERMS - depth - 1) terms,
        # each >= d+1 and < limit_max.
        # The maximum possible sum with k terms each < limit_max is bounded by
        # k/limit_max (very loose). Better: the terms must be >= d+1, so max sum
        # is roughly k/(d+1) for small k, but this is also loose.
        
        # Key pruning: if rem_new > 0 and (MAX_TERMS - depth - 1) == 0, fail.
        # Also: the smallest denominator in remaining is d+1, so max sum with
        # k terms is at most k/(d+1) (very loose upper bound on what's achievable).
        # Actually that's an upper bound on the sum, not lower. We need a LOWER bound
        # on what's achievable to prune: the MINIMUM sum with k terms each >= d+1
        # and < limit_max is... well, it can be arbitrarily small. So no easy lower bound.
        
        # Better pruning: if rem_new > 0, we need at least one more term, so
        # MAX_TERMS - depth - 1 >= 1, i.e., depth <= MAX_TERMS - 2.
        # Already checked above.
        
        # Another bound: if rem_new > 0, the next denominator d2 must satisfy
        # d2 >= ceil(1/rem_new). If ceil(1/rem_new) >= limit_max, fail.
        # This is handled by the recursive call.
        
        # Iterate d from min_d_eff to limit_max - 1
        # Optimization: skip d if d is in used
        for d in range(min_d_eff, limit_max):
            if d in used:
                continue
            rem_new = rem - Fraction(1, d)
            if rem_new < 0:
                # Since d is increasing, 1/d is decreasing, rem_new is increasing.
                # So if rem_new < 0, it will become >= 0 for larger d.
                # But we want rem_new >= 0 (or we could allow negative and handle it,
                # but standard Egyptian fractions use positive remainders).
                # Actually if rem_new < 0, that means 1/d > rem, which is fine
                # only if we're not at the last term. But for simplicity, let's
                # continue searching (don't break).
                # Wait: if 1/d > rem, then rem - 1/d < 0. For the next term,
                # we'd need a negative reciprocal, which is not allowed.
                # So we MUST have 1/d <= rem, i.e., d >= ceil(1/rem).
                # But we already set min_d_eff = max(min_d, ceil(1/rem)).
                # So rem_new should be >= 0 for all d >= min_d_eff.
                # Let me double-check: ceil(1/rem) means d >= ceil(1/rem) => 1/d <= 1/ceil(1/rem) <= rem.
                # Yes, so rem_new >= 0 always. The check is redundant but safe.
                continue
            if rem_new == 0:
                return [d]
            
            # Additional pruning: if rem_new > 0, we need more terms.
            # The next min_d is d+1. Check if ceil(1/rem_new) < limit_max.
            next_min = d + 1
            next_min_eff = max(next_min, (rem_new.denominator + rem_new.numerator - 1) // rem_new.numerator)
            if next_min_eff >= limit_max:
                continue
            
            res = _search_bounded(rem_new, d + 1, depth + 1, limit_max, used | {d})
            if res is not None:
                return [d] + res
        return None

    # Direct 3-term enumeration: for each a, solve 1/b + 1/c = rem where rem = 5/31 - 1/a
    # This gives (n*b - d)(n*c - d) = d^2 where rem = n/d
    # Much faster than general recursive search
    
    for a in range(7, best_max):
        if time.time() > deadline:
            break
        rem = TARGET - Fraction(1, a)
        if rem <= 0:
            continue
        n, d = rem.numerator, rem.denominator
        # Solve n(b+c) = d*bc => (nb-d)(nc-d) = d^2
        # Let x = nb - d, y = nc - d, then xy = d^2, x < y (since b < c)
        # b = (x+d)/n must be integer > a, c = (y+d)/n must be integer > b
        d_sq = d * d
        # Find all factor pairs (x, y) with x*y = d_sq, x <= y
        # x must satisfy: x > 0, (x+d) % n == 0, b = (x+d)//n > a
        # y = d_sq // x, (y+d) % n == 0, c = (y+d)//n > b
        
        # Iterate x from 1 to sqrt(d_sq)
        x_max = int(d_sq ** 0.5)
        for x in range(1, x_max + 1):
            if d_sq % x != 0:
                continue
            y = d_sq // x
            if x > y:
                break
            if (x + d) % n != 0:
                continue
            b = (x + d) // n
            if b <= a:
                continue
            if (y + d) % n != 0:
                continue
            c = (y + d) // n
            if c <= b:
                continue
            if c >= best_max:
                continue
            # Found a 3-term solution [a, b, c] with max = c
            best_max = c
            best_list = [a, b, c]
            break  # Found best c for this a, move to next a
    
    # Now try 4-term solutions that might have smaller max than current best_max
    # For each a, b, solve 1/c + 1/d = rem where rem = 5/31 - 1/a - 1/b
    for a in range(7, min(best_max, 50)):  # Limit a to keep search feasible
        if time.time() > deadline:
            break
        for b in range(a + 1, min(best_max, 200)):
            if time.time() > deadline:
                break
            rem = TARGET - Fraction(1, a) - Fraction(1, b)
            if rem <= 0:
                continue
            n, d = rem.numerator, rem.denominator
            d_sq = d * d
            x_max = int(d_sq ** 0.5)
            for x in range(1, x_max + 1):
                if d_sq % x != 0:
                    continue
                y = d_sq // x
                if x > y:
                    break
                if (x + d) % n != 0:
                    continue
                c = (x + d) // n
                if c <= b:
                    continue
                if (y + d) % n != 0:
                    continue
                dd = (y + d) // n
                if dd <= c:
                    continue
                if dd >= best_max:
                    continue
                best_max = dd
                best_list = [a, b, c, dd]
                break
    
    return best_list

# EVOLVE-BLOCK-END
