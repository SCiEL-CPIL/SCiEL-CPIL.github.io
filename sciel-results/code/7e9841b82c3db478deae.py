"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

_DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
_N = len(_DIVISORS)
_MAX_VAL = 1000000

def _lcm(a, b):
    return a * b // math.gcd(a, b)

_LCM = 1
for d in _DIVISORS:
    _LCM = _lcm(_LCM, d)

def _can_assign_fast(L, R, divisors):
    """Check if we can assign distinct multiples in [L, L+R-1] for each divisor.
    Returns the assignment list or None."""
    n = len(divisors)
    candidates = []
    for i in range(n):
        d = divisors[i]
        k_start = (L + d - 1) // d
        k_end = (L + R - 1) // d
        if k_start > k_end:
            return None
        cand = list(range(k_start * d, k_end * d + 1, d))
        candidates.append(cand)
    
    # MRV heuristic: process divisors with fewest candidates first
    order = sorted(range(n), key=lambda i: len(candidates[i]))
    assigned = [None] * n
    used = set()
    
    def backtrack(idx):
        if idx == n:
            return True
        i = order[idx]
        for val in candidates[i]:
            if val not in used:
                used.add(val)
                assigned[i] = val
                if backtrack(idx + 1):
                    return True
                used.remove(val)
                assigned[i] = None
        return False
    
    if backtrack(0):
        return assigned
    return None

def construct():
    deadline = time.time() + 85
    divisors = _DIVISORS
    n = len(divisors)
    
    # Search over R from small to large
    # For each R, check all L in one LCM period
    for R in range(1, 1000):
        if time.time() > deadline:
            break
        
        # Early feasibility check: each divisor must have at least one multiple in [L, L+R-1]
        # For a given L, divisor d_i has a multiple in [L, L+R-1] iff 
        # there exists k with L <= k*d_i <= L+R-1
        # This is equivalent to: floor((L+R-1)/d_i) >= ceil(L/d_i)
        
        for L in range(1, _LCM + 1):
            if time.time() > deadline:
                break
            
            assigned = _can_assign_fast(L, R, divisors)
            if assigned is not None:
                if max(assigned) <= _MAX_VAL:
                    return [int(assigned[i]) for i in range(n)]
    
    # Fallback
    return [int(d) for d in divisors]

# EVOLVE-BLOCK-END
