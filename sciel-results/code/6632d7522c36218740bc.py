"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import product

def _lcm(a, b):
    return a // gcd(a, b) * b

def _compute_lcm(lst):
    r = 1
    for x in lst:
        r = _lcm(r, x)
    return r

_DIVS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
_LCM = _compute_lcm(_DIVS)
_N = len(_DIVS)
_MAX_VAL = 1000000

def _find_multiple_in_window(L, W, d):
    """Find the smallest multiple of d in [L, L+W], or None if no such multiple."""
    rem = L % d
    if rem == 0:
        m = L
    else:
        m = L + (d - rem)
    if m <= L + W:
        return m
    return None

def _solve_for_W(W):
    """
    Check if there exists a window of width W with distinct multiples.
    Returns (True, result_list) if found, else (False, None).
    """
    rigid_divs = [(d, i) for i, d in enumerate(_DIVS) if d > W]
    flex_divs = [(d, i) for i, d in enumerate(_DIVS) if d <= W]
    
    for L in range(_LCM):
        # 1. Handle rigid divisors
        used = {}
        rigid_vals = {}
        valid = True
        
        for d, idx in rigid_divs:
            m = _find_multiple_in_window(L, W, d)
            if m is None:
                valid = False
                break
            if m in used:
                valid = False
                break
            used[m] = idx
            rigid_vals[idx] = m
        
        if not valid:
            continue
        
        # 2. Handle flexible divisors
        flex_candidates = {}
        for d, idx in flex_divs:
            cands = []
            rem = L % d
            if rem == 0:
                m = L
            else:
                m = L + (d - rem)
            
            while m <= L + W:
                if m not in used:
                    cands.append(m)
                m += d
            if not cands:
                valid = False
                break
            flex_candidates[idx] = cands
        
        if not valid:
            continue
        
        # 3. Backtracking for flexible divisors
        flex_items = sorted(flex_candidates.items(), key=lambda x: len(x[1]))
        
        def backtrack(i, used_so_far, assignment):
            if i == len(flex_items):
                return True
            
            idx, cands = flex_items[i]
            for val in cands:
                if val not in used_so_far:
                    used_so_far.add(val)
                    assignment[idx] = val
                    if backtrack(i + 1, used_so_far, assignment):
                        return True
                    del assignment[idx]
                    used_so_far.remove(val)
            return False
        
        assignment = dict(rigid_vals)
        used_so_far = set(used.keys())
        
        if backtrack(0, used_so_far, assignment):
            result = [0] * _N
            for idx, val in assignment.items():
                result[idx] = val
            
            if max(result) > _MAX_VAL:
                continue
            
            return (True, result)
    
    return (False, None)

def construct():
    t_start = time.time()
    deadline = t_start + 85.0
    
    best_result = None
    best_W = None
    
    for W in range(1, 34):
        if time.time() > deadline:
            break
        
        ok, res = _solve_for_W(W)
        if ok:
            best_result = res
            best_W = W
            break
    
    if best_result is not None:
        return [int(x) for x in best_result]
    
    return [int(d) for d in _DIVS]

# EVOLVE-BLOCK-END
