"""Optimize over geometric-progressions U = {floor(b^k)} to maximize the
sumset/difference-set exponent lower bound."""
import math
import time
import itertools

# EVOLVE-BLOCK-START
def _score_of(U):
    """Return the raw lower-bound score for a concrete set U (must contain 0)."""
    if not U:
        return 0.0
    u = sorted(set(U))
    if 0 not in u:
        u.append(0)
    n = len(u)
    mx = u[-1]
    if mx <= 0:
        return 0.0
    # difference set U - U
    diffs = set()
    for a in u:
        for b in u:
            diffs.add(a - b)
    # sum set U + U
    sums = set()
    for a in u:
        for b in u:
            sums.add(a + b)
    denom = 2.0 * mx + 1.0
    if denom <= 1.0:
        return 0.0
    ratio = len(diffs) / float(len(sums))
    if ratio <= 0.0:
        return 0.0
    return 1.0 + (math.log(ratio) / math.log(denom))

def _build_geo(b, m, include_zero=True, start_k=1):
    """Construct U = {floor(b^k) | k=start_k..start_k+m-1} plus optionally 0."""
    U = []
    if include_zero:
        U.append(0)
    for k in range(start_k, start_k + m):
        v = int(math.floor(b ** k))
        if v >= 0:
            U.append(v)
    # Remove duplicates while preserving order
    seen = set()
    result = []
    for x in U:
        if x not in seen:
            seen.add(x)
            result.append(x)
    return result

def _ternary_search_b(m, b_lo, b_hi, tol=0.001, max_iter=50):
    """Ternary search over b in [b_lo, b_hi] to maximize score for fixed m."""
    lo = b_lo
    hi = b_hi
    best_b = b_lo
    best_s = -1.0
    
    for _ in range(max_iter):
        if hi - lo < tol:
            break
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        
        U1 = _build_geo(m1, m)
        U2 = _build_geo(m2, m)
        
        s1 = _score_of(U1)
        s2 = _score_of(U2)
        
        if s1 > best_s:
            best_s = s1
            best_b = m1
        if s2 > best_s:
            best_s = s2
            best_b = m2
        
        if s1 < s2:
            lo = m1
        else:
            hi = m2
    
    # Final evaluation at the best point found
    U_best = _build_geo(best_b, m)
    s_best = _score_of(U_best)
    if s_best > best_s:
        best_s = s_best
    
    return best_s, best_b, U_best

def construct():
    deadline = time.time() + 100.0
    best_score = 0.0
    best_set = [0, 1, 2, 4, 8]
    
    # First, evaluate the known good set
    s0 = _score_of(best_set)
    if s0 > best_score:
        best_score = s0
    
    # Search over m (length of progression) and b (base)
    # For each m, do a coarse grid search over b, then refine with ternary search
    for m in range(2, 15):
        if time.time() > deadline:
            break
        
        # Coarse grid over b
        b_vals = []
        # Start b from just above 1 up to about 5
        b_start = 1.01
        b_end = 5.0
        b_step = 0.05
        
        b = b_start
        coarse_best_s = -1.0
        coarse_best_b = b_start
        coarse_best_U = None
        
        while b <= b_end:
            if time.time() > deadline:
                break
            U = _build_geo(b, m)
            if len(U) < 2:
                b += b_step
                continue
            s = _score_of(U)
            if s > coarse_best_s:
                coarse_best_s = s
                coarse_best_b = b
                coarse_best_U = U
            b += b_step
        
        # Refine around the coarse best using ternary search
        if coarse_best_U is not None:
            b_lo = max(1.01, coarse_best_b - 0.1)
            b_hi = min(5.0, coarse_best_b + 0.1)
            s_ref, b_ref, U_ref = _ternary_search_b(m, b_lo, b_hi, tol=0.001, max_iter=30)
            
            if s_ref > best_score:
                best_score = s_ref
                best_set = U_ref
        
        # Also try a few specific promising b values for this m
        for b_try in [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.2, 2.5, 3.0]:
            if time.time() > deadline:
                break
            U = _build_geo(b_try, m)
            if len(U) < 2:
                continue
            s = _score_of(U)
            if s > best_score:
                best_score = s
                best_set = U
    
    return best_set
# EVOLVE-BLOCK-END
