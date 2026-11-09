"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import ceil

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000
PERIOD = 30030  # lcm of all divisors

def _find_assignment(L, W, divisors):
    """Find distinct multiples for each divisor in [L, L+W].
    Returns list of chosen multiples or None."""
    n = len(divisors)
    choices = []
    for d in divisors:
        # multiples of d in [L, L+W]
        first = ((L + d - 1) // d) * d
        mults = []
        m = first
        while m <= L + W:
            mults.append(m)
            m += d
        if not mults:
            return None
        choices.append(mults)
    
    # Backtracking
    assigned = [None] * n
    used = set()
    
    def backtrack(idx):
        if idx == n:
            return True
        for m in choices[idx]:
            if m not in used:
                used.add(m)
                assigned[idx] = m
                if backtrack(idx + 1):
                    return True
                used.remove(m)
                assigned[idx] = None
        return False
    
    if backtrack(0):
        return assigned
    return None

def _solve():
    start_time = time.time()
    deadline = start_time + 80  # safety margin
    
    best = None
    best_W = None
    
    # Search W from 1 upward
    for W in range(1, 200):
        if time.time() > deadline:
            break
        for L in range(0, PERIOD):
            if time.time() > deadline:
                break
            result = _find_assignment(L, W, DIVISORS)
            if result is not None:
                # Check all values <= MAX_VAL and distinct (already ensured)
                if max(result) <= MAX_VAL:
                    best = result
                    best_W = W
                    # Found a solution for this W; check if we can do better
                    # but we want minimal W, so return immediately
                    return result
    return best

def construct():
    result = _solve()
    if result is None:
        # Fallback: just use the divisors themselves
        return [int(d) for d in DIVISORS]
    return [int(x) for x in result]
# EVOLVE-BLOCK-END
