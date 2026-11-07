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

# Precompute for each divisor i and each residue r in [0, LCM):
# the offsets p in [0, R-1] such that (L+p) % d_i == 0 where L % LCM = r
# But R varies, so instead: for a given L and R, the candidate offsets for divisor i
# are p in [0, R-1] such that (L+p) % d_i == 0, i.e., p ≡ (-L) % d_i (mod d_i)

# Faster matching using bitmasks
def _can_assign_fast(L, R, divisors):
    """Check if we can assign distinct multiples in [L, L+R-1] for each divisor.
    Returns the assignment list or None."""
    n = len(divisors)
    # For each divisor, compute candidate values as a bitmask over offsets [0, R-1]
    # But we need distinct VALUES, not offsets. Two different divisors could have
    # the same value at different offsets only if the values coincide.
    # Actually, values are L + p, so distinct offsets give distinct values.
    # So we need to pick distinct offsets p_i for each divisor i.
    
    # For divisor i, valid offsets p satisfy: p ≡ (-L) % d_i (mod d_i) and 0 <= p < R
    # Let's compute the set of valid offsets for each divisor
    
    valid_offsets = []
    for i in range(n):
        d = divisors[i]
        target = (-L) % d
        # Find all p in [0, R-1] such that p % d == target
        offsets = []
        if target < R:
            p = target
            while p < R:
                offsets.append(p)
                p += d
        if not offsets:
            return None
        valid_offsets.append(offsets)
    
    # Now we need to assign distinct offsets to each divisor.
    # This is a bipartite matching problem.
    # Use backtracking with MRV.
    order = sorted(range(n), key=lambda i: len(valid_offsets[i]))
    assigned_offset = [-1] * n
    used_offsets = [False] * R
    
    def backtrack(idx):
        if idx == n:
            return True
        i = order[idx]
        for p in valid_offsets[i]:
            if not used_offsets[p]:
                used_offsets[p] = True
                assigned_offset[i] = p
                if backtrack(idx + 1):
                    return True
                used_offsets[p] = False
                assigned_offset[i] = -1
        return False
    
    if backtrack(0):
        return [L + assigned_offset[i] for i in range(n)]
    return None

def construct():
    deadline = time.time() + 85
    divisors = _DIVISORS
    n = len(divisors)
    
    # Search over R from 1 upward
    for R in range(1, 5000):
        if time.time() > deadline:
            break
        
        # For this R, iterate L over one LCM period
        for L in range(1, _LCM + 1):
            if time.time() > deadline:
                break
            
            assigned = _can_assign_fast(L, R, divisors)
            if assigned is not None:
                if max(assigned) <= _MAX_VAL:
                    return [int(x) for x in assigned]
    
    # Fallback
    return [int(d) for d in divisors]

# EVOLVE-BLOCK-END
