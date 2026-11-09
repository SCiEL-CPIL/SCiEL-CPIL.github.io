"""
Combines the structural search loop of the current program with the 
backtracking assignment logic from Inspiration 1 to handle distinctness 
constraints more effectively than the greedy 'smallest multiple' approach.
"""
import time
from math import gcd
from functools import reduce

# EVOLVE-BLOCK-START
_DIVS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
_MAXV = 1000000
_DEADLINE = 80.0  # wall-clock seconds (90s kill, leave margin)

def _lcm(a, b):
    return a // gcd(a, b) * b

# Precompute LCM of all divisors
_LCM_ALL = reduce(_lcm, _DIVS)

def _try_assign(L, W):
    """
    Try to assign distinct multiples of each divisor within [L, L+W].
    Returns list of assigned values (aligned with _DIVS order) or None.
    Uses backtracking with MRV (Minimum Remaining Values) heuristic.
    """
    # Precompute candidate multiples for each divisor
    candidates = []
    for d in _DIVS:
        cands = []
        # First multiple >= L
        k0 = (L + d - 1) // d
        # Last multiple <= L+W
        k1 = (L + W) // d
        if k0 <= k1:
            # Generate candidates
            # If the window is large, this list could be big, but W is expected to be small
            cands = [k * d for k in range(k0, k1 + 1)]
        if not cands:
            return None
        candidates.append(cands)
    
    # Order divisors by number of candidates (MRV)
    # We need to map back to original indices
    indices = list(range(len(_DIVS)))
    indices.sort(key=lambda i: len(candidates[i]))
    
    assigned = [None] * len(_DIVS)
    used = set()
    
    def backtrack(idx):
        if idx == len(indices):
            return True
        i = indices[idx]
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
        return assigned
    return None

def construct():
    t0 = time.time()
    
    # Lower bound on W: 10 distinct integers in a closed interval of length W
    # implies W >= 9.
    # Due to divisibility constraints, W will likely be larger.
    # We start from 9 and go up.
    
    # Key optimization: The feasibility pattern is periodic with period _LCM_ALL.
    # For a fixed W, if no solution exists in L ∈ [1, min(_LCM_ALL, _MAXV - W)],
    # then no solution exists for that W at all.
    
    for W in range(9, 200):
        if time.time() - t0 > _DEADLINE:
            break
        
        # Only need to check L in one period
        L_limit = min(_LCM_ALL, _MAXV - W)
        if L_limit < 1:
            continue
        
        found = False
        for L in range(1, L_limit + 1):
            if time.time() - t0 > _DEADLINE:
                break
            reps = _try_assign(L, W)
            if reps is not None:
                # Found optimal W and a valid assignment
                return [int(x) for x in reps]
        
        # If we finished scanning L for this W without success, try next W
    
    # Fallback: Random search or return divisors
    import random
    random.seed(42)
    best_reps = None
    best_W = float('inf')
    
    # Try random L for a few W values if we have time
    for W in range(9, 100):
        if time.time() - t0 > _DEADLINE - 5:
            break
        for _ in range(1000):
            if time.time() - t0 > _DEADLINE - 5:
                break
            L = random.randint(1, max(1, _MAXV - W))
            reps = _try_assign(L, W)
            if reps is not None:
                if W < best_W:
                    best_reps = reps
                    best_W = W
    
    if best_reps is not None:
        return [int(x) for x in best_reps]
    
    # Ultimate fallback
    return [int(d) for d in _DIVS]
# EVOLVE-BLOCK-END
