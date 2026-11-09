"""Find 10 distinct multiples of divisors [6,10,14,15,21,22,26,33,35,39]
in a minimal-width window [L, L+W], each <= 1e6."""
import time
from math import gcd
from functools import reduce

_DIVS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
_MAXV = 1000000
_DEADLINE = 80.0

def _lcm(a, b):
    return a // gcd(a, b) * b

_LCM_ALL = reduce(_lcm, _DIVS)

def _multiples_in_window(L, W, d):
    """Return list of multiples of d in [L, L+W]."""
    if L > _MAXV:
        return []
    hi = min(L + W, _MAXV)
    # first multiple >= L
    k0 = (L + d - 1) // d
    k1 = hi // d
    if k0 > k1:
        return []
    return [k * d for k in range(k0, k1 + 1)]

def _try_assign(L, W):
    """Try to assign distinct multiples of each divisor within [L, L+W].
    Returns list of assigned values or None."""
    # Precompute candidate multiples for each divisor
    candidates = []
    for d in _DIVS:
        cands = _multiples_in_window(L, W, d)
        if not cands:
            return None
        candidates.append(cands)
    
    # Backtracking: assign in order of fewest candidates first (MRV heuristic)
    order = sorted(range(len(_DIVS)), key=lambda i: len(candidates[i]))
    assigned = [None] * len(_DIVS)
    used = set()
    
    def backtrack(idx):
        if idx == len(order):
            return True
        i = order[idx]
        for val in candidates[i]:
            if val not in used:
                used.add(val)
                assigned[i] = val
                if backtrack(idx + 1):
                    return True
                used.discard(val)
                assigned[i] = None
        return False
    
    if backtrack(0):
        return [assigned[i] for i in range(len(_DIVS))]
    return None

def construct():
    t0 = time.time()
    best_reps = None
    best_W = float('inf')
    
    # Lower bound: 10 distinct ints need window >= 9
    # But divisibility constraints force larger. Start from a reasonable bound.
    # The largest divisor is 39, so a window of at least 38 might be needed
    # to guarantee a multiple of 39, but we can do better with smarter L.
    
    # Strategy: for W from lower to upper, scan L over a structured range.
    # For a given W, L must satisfy: for each d, there exists a multiple of d in [L, L+W].
    # This means L mod d must be in a specific range. We can enumerate valid L residues.
    
    # Phase 1: Try small W with a scan of L
    # To be efficient, note that the pattern repeats with period LCM of divisors.
    # But LCM is huge (~10^8+), so we can't enumerate all L.
    
    # Instead: for each W, try L values that are "aligned" — e.g., L such that
    # L is close to a multiple of several divisors.
    
    # Simpler robust approach: scan L from 1 to some limit for each W.
    # Limit L to 1..200000 to keep it fast, and also try random L for larger W.
    
    for W in range(9, 300):
        if time.time() - t0 > _DEADLINE:
            break
        
        # For this W, scan L over a range
        # We can restrict L: the window [L, L+W] must be within [1, _MAXV]
        # So L in [1, _MAXV - W]
        # But scanning all L is too slow. Use a step or structured sampling.
        
        # Key insight: if L works, then L + k*LCM_ALL also works (same residues).
        # So we only need to check L in [1, min(LCM_ALL, _MAXV - W)] but LCM_ALL is huge.
        
        # Alternative: for each divisor d, the condition "there's a multiple of d in [L, L+W]"
        # is equivalent to: L mod d ∈ {0} ∪ [d - W, d - 1] (if W < d) or always true (if W >= d-1).
        # For W >= 38 (max divisor - 1), every L works for the existence condition.
        # For smaller W, we need to check residues.
        
        # Let's just scan L in steps. For W < 40, use step 1 up to a limit.
        # For W >= 40, every L has at least one multiple of each d, so we just need
        # to find L where the multiples can be made distinct.
        
        if W >= 38:
            # Every L has multiples of all divisors in [L, L+W]
            # Just need distinctness. Try a few L values.
            l_range = range(1, min(_MAXV - W + 1, 5000))
            for L in l_range:
                if time.time() - t0 > _DEADLINE:
                    break
                reps = _try_assign(L, W)
                if reps is not None:
                    best_reps = reps
                    best_W = W
                    break
            if best_reps is not None:
                break
        else:
            # W < 38: need to check which L have multiples of all divisors
            # Scan L from 1 to a limit
            l_limit = min(_MAXV - W + 1, 100000)
            for L in range(1, l_limit + 1):
                if time.time() - t0 > _DEADLINE:
                    break
                # Quick check: does each divisor have a multiple in [L, L+W]?
                ok = True
                for d in _DIVS:
                    k0 = (L + d - 1) // d
                    k1 = (L + W) // d
                    if k0 > k1:
                        ok = False
                        break
                if not ok:
                    continue
                reps = _try_assign(L, W)
                if reps is not None:
                    best_reps = reps
                    best_W = W
                    break
            if best_reps is not None:
                break
    
    # If not found, do a broader random search
    if best_reps is None:
        import random
        random.seed(42)
        for _ in range(100000):
            if time.time() - t0 > _DEADLINE:
                break
            W = random.randint(40, 500)
            L = random.randint(1, _MAXV - W)
            reps = _try_assign(L, W)
            if reps is not None:
                if W < best_W:
                    best_reps = reps
                    best_W = W
                    break
    
    # Fallback: return the divisors themselves
    if best_reps is None:
        best_reps = list(_DIVS)
    
    return [int(x) for x in best_reps]
