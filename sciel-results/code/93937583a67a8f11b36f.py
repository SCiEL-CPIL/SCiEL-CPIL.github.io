"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

_MAX_DEN = 10000000


def _max_fill_float(last, count):
    # Approximate max reciprocal sum with `count` distinct denominators > last.
    s = 0.0
    for j in range(1, count + 1):
        s += 1.0 / (last + j)
    return s


def _search(k, deadline):
    # Backtracking for exactly k distinct denominators >= 5 summing to 1.
    # Uses integer arithmetic (num/den) for exactness, float for pruning.
    
    def rec(remaining, last, num, den):
        if time.time() > deadline:
            return None
        
        # Prune: sum already exceeds 1
        if num > den:
            return None
            
        if remaining == 0:
            return None if num != den else []
            
        # Prune: even taking the smallest available denominators, we can't reach 1
        max_possible = _max_fill_float(last, remaining)
        if (num / den) + max_possible < 1.0:
            return None
            
        if remaining == 1:
            # We need 1/d = 1 - num/den = (den - num) / den
            # So d = den / (den - num)
            diff = den - num
            if diff <= 0:
                return None
            if den % diff == 0:
                d = den // diff
                if d > last and d >= 5 and d <= _MAX_DEN:
                    return [d]
            return None

        # remaining >= 2
        diff = den - num
        if diff <= 0:
            return None
            
        # d must satisfy 1/d <= diff/den => d >= den/diff
        d_low = max(last + 1, (den + diff - 1) // diff)
        
        d = d_low
        guard = 0
        while d <= _MAX_DEN:
            guard += 1
            if guard > 100000 or time.time() > deadline:
                return None
                
            # Update num, den with 1/d
            new_num = num * d + den
            new_den = den * d
            
            # Reduce by GCD
            g = math.gcd(new_num, new_den)
            if g > 1:
                new_num //= g
                new_den //= g
                
            # Check if we can still reach 1
            if new_num > new_den:
                d += 1
                continue
                
            # Prune: check if max possible sum from next steps is enough
            if (new_num / new_den) + _max_fill_float(d, remaining - 1) < 1.0:
                break
                
            res = rec(remaining - 1, d, new_num, new_den)
            if res is not None:
                return [d] + res
                
            d += 1
            
        return None

    return rec(k, 4, 0, 1)


def _fallback():
    # Guaranteed-valid baseline.
    # If search fails, return a known valid solution for k=8 or larger.
    # A valid solution for k=8 is: [5, 6, 7, 8, 9, 10, 11, 12] is NOT valid.
    # We rely on the search finding k=8.
    raise RuntimeError("no baseline representation found")


def construct():
    # k < 8 is impossible.
    deadline = time.time() + 50.0
    for k in range(8, 257):
        sol = _search(k, deadline)
        if sol is not None:
            return sol
    return _fallback()
# EVOLVE-BLOCK-END
