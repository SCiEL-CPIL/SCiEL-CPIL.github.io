"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

_DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
_LCM = 30030
_MAX_VAL = 1000000
_N = 10

# Precompute residues for each divisor
_residues = []
for d in _DIVISORS:
    res = []
    for p in range(0, _LCM, d):
        res.append(p)
    _residues.append(res)

# Sort by scarcity (fewest candidates first)
_indices = sorted(range(_N), key=lambda i: len(_residues[i]))
_sorted_divisors = [_DIVISORS[i] for i in _indices]
_sorted_residues = [_residues[i] for i in _indices]

def _check_window(W):
    """
    Check if there is a window of length W that allows distinct assignment.
    Returns (offset, assignment_list) where assignment_list is in original divisor order.
    """
    for S in range(_LCM):
        # Precompute candidates for each divisor in this window
        candidates = []
        for idx in _indices:
            d = _DIVISORS[idx]
            res_list = _residues[idx]
            cands = []
            for p in res_list:
                if (p - S) % _LCM <= W:
                    cands.append(p)
            if not cands:
                break
            candidates.append((idx, cands))
        else:
            # All divisors have at least one candidate, try backtracking
            used = set()
            assignment = [None] * _N
            
            def backtrack(pos):
                if pos == _N:
                    return True
                
                orig_idx, cands = candidates[pos]
                for p in cands:
                    if p not in used:
                        used.add(p)
                        assignment[orig_idx] = p
                        if backtrack(pos + 1):
                            return True
                        used.remove(p)
                        assignment[orig_idx] = None
                return False
            
            if backtrack(0):
                return S, assignment
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    best_W = None
    best_S = None
    best_assignment = None
    
    lo = 1
    hi = _LCM
    
    # Binary search on W
    while lo < hi:
        if time.time() > deadline:
            break
        mid = (lo + hi) // 2
        res = _check_window(mid)
        if res is None:
            if time.time() > deadline:
                break
            lo = mid + 1
        else:
            S, assignment = res
            best_W = mid
            best_S = S
            best_assignment = assignment
            hi = mid
    
    if best_W is None:
        return list(_DIVISORS)
    
    # Construct actual integers
    min_p = min(best_assignment)
    max_p = max(best_assignment)
    
    S_base = 0
    min_needed = 1 - min_p
    max_allowed = 1000000 - max_p
    
    if min_needed <= 0:
        S_base = 0
    else:
        S_base = ((min_needed + _LCM - 1) // _LCM) * _LCM
        
    if S_base > max_allowed:
        return list(_DIVISORS)
        
    result = [S_base + p for p in best_assignment]
    
    # Verify
    for i, val in enumerate(result):
        if val <= 0 or val > 1000000:
            return list(_DIVISORS)
        if val % _DIVISORS[i] != 0:
            return list(_DIVISORS)
    if len(set(result)) != _N:
        return list(_DIVISORS)
        
    return result

# EVOLVE-BLOCK-END
