"""Improved program: LCM-periodic window search with exact backtracking."""
# EVOLVE-BLOCK-START
import math
import time

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000
N = len(DIVISORS)

def _lcm(a, b):
    return a * b // math.gcd(a, b)

# Compute LCM of all divisors (period of divisibility pattern)
_LCM = 1
for d in DIVISORS:
    _LCM = _lcm(_LCM, d)

# Precompute: for each position in [0, LCM-1], which divisors divide (L + pos)
# We'll use this to quickly check candidates in a window.

def _build_pattern():
    """Build a bitmask for each residue r in [0, LCM-1] indicating which divisors divide numbers with that residue."""
    # pattern[r] = bitmask of divisors d such that d | (L + r) for any L multiple of LCM
    # Actually: d | (L + r) iff d | r (when L is multiple of LCM, and LCM is multiple of d)
    # Since LCM is a multiple of every d, L % d == 0, so (L + r) % d == r % d.
    # So d | (L + r) iff d | r.
    pattern = [0] * _LCM
    for r in range(_LCM):
        mask = 0
        for i, d in enumerate(DIVISORS):
            if r % d == 0:
                mask |= (1 << i)
        pattern[r] = mask
    return pattern

_PATTERN = _build_pattern()

def _solve_window(L, W, deadline):
    """Try to find distinct multiples in [L, L+W-1] for each divisor.
    Returns list of 10 ints or None."""
    # For each divisor i, candidate values in [L, L+W-1] that are multiples of DIVISORS[i]
    # Use the pattern to quickly identify valid residues
    
    # Build candidate lists
    candidates = []
    for i, d in enumerate(DIVISORS):
        # First multiple of d >= L
        start = ((L + d - 1) // d) * d
        cands = []
        v = start
        while v < L + W:
            cands.append(v)
            v += d
        candidates.append(cands)
    
    # Check if any divisor has no candidates
    for i in range(N):
        if not candidates[i]:
            return None
    
    # Backtracking: assign distinct values to each divisor
    used = set()
    result = [0] * N
    
    # Order divisors by fewest candidates first (MRV heuristic)
    order = sorted(range(N), key=lambda i: len(candidates[i]))
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == N:
            return True
        i = order[idx]
        for v in candidates[i]:
            if v not in used:
                used.add(v)
                result[i] = v
                if backtrack(idx + 1):
                    return True
                used.remove(v)
        return False
    
    if backtrack(0):
        return result[:]
    return None

def construct():
    deadline = time.time() + 80  # 10s safety margin
    
    best = None
    best_width = float('inf')
    
    # Strategy 1: Try small windows starting from various L positions
    # The window width W must be at least N-1 = 9 (since we need 10 distinct integers)
    # Start with W = 9 and increase
    
    # First pass: try W from 9 upward, scanning L over one LCM period
    # For each W, scan L in [0, LCM-1] (since pattern is periodic with period LCM)
    for W in range(9, 200):
        if time.time() > deadline:
            break
        found = False
        # Scan L over one period
        for L in range(0, _LCM, 1):
            if time.time() > deadline:
                break
            # Check if window [L, L+W-1] is within [1, MAX_VAL]
            if L < 1 or L + W - 1 > MAX_VAL:
                continue
            res = _solve_window(L, W, deadline)
            if res is not None:
                # Verify
                assert len(set(res)) == N
                assert all(1 <= v <= MAX_VAL for v in res)
                assert all(res[i] % DIVISORS[i] == 0 for i in range(N))
                width = max(res) - min(res)
                if width < best_width:
                    best_width = width
                    best = res[:]
                found = True
                break  # Found a solution for this W, move to next W
        
        if found:
            # We found a solution for this W; try slightly larger W from other L positions
            # But first, check if we can do better with the same W from other L
            # Actually, let's continue scanning for smaller effective width
            pass
    
    # Strategy 2: If we have a solution, try to improve by local search
    if best is not None:
        # Try shifting the window
        cur_min = min(best)
        cur_max = max(best)
        cur_width = cur_max - cur_min
        
        # Try to find a tighter solution by scanning nearby L values
        for delta_L in range(-50, 51):
            if time.time() > deadline:
                break
            L = cur_min + delta_L
            W = cur_width + 1  # Slightly larger window to allow rearrangement
            if L < 1 or L + W - 1 > MAX_VAL:
                continue
            res = _solve_window(L, W, deadline)
            if res is not None:
                width = max(res) - min(res)
                if width < best_width:
                    best_width = width
                    best = res[:]
    
    if best is None:
        # Fallback: return the divisors themselves (valid but not optimal)
        return DIVISORS[:]
    
    return best
# EVOLVE-BLOCK-END
