"""Pack distinct multiples of a fixed integer set - minimize the enclosing span."""
import time

# EVOLVE-BLOCK-START
_DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
_MAXV = 1000000
_N = len(_DIVISORS)
# LCM of all divisors: 2*3*5*7*11*13 = 30030
_PERIOD = 30030
_DEADLINE = time.time() + 85  # safety margin vs 90 s wall-clock
_BEST = list(_DIVISORS)  # always-valid fallback
_BEST_SPAN = max(_BEST) - min(_BEST)


def _check_window_fast(L, S):
    """Check if we can assign distinct multiples of each divisor in [L, L+S].
    Returns list of representatives if possible, else None.
    Optimized with early pruning and MRV heuristic.
    """
    U = L + S
    # For each divisor, find all multiples in [L, U]
    cands = []
    for d in _DIVISORS:
        k = (L + d - 1) // d
        m = k * d
        if m > U:
            return None  # No candidates for this divisor
        # Count candidates first for quick rejection
        cnt = (U - m) // d + 1
        if cnt == 0:
            return None
        # Build candidate list
        multiples = [m + j * d for j in range(cnt)]
        cands.append(multiples)
    
    # Use backtracking with MRV (minimum remaining values) heuristic
    # Order divisors by number of candidates (ascending)
    order = sorted(range(_N), key=lambda i: len(cands[i]))
    
    # If any divisor has exactly 1 candidate, check for conflicts early
    used = set()
    reps = [0] * _N
    
    def bt(pos):
        if pos == _N:
            return True
        i = order[pos]
        for val in cands[i]:
            if val not in used:
                used.add(val)
                reps[i] = val
                if bt(pos + 1):
                    return True
                used.remove(val)
        return False
    
    if bt(0):
        return reps
    return None


def _generate_candidate_Ls(S):
    """Generate only L values where at least one divisor has a multiple in [L, L+S].
    For each divisor d, multiples occur at positions that are multiples of d.
    A window [L, L+S] contains a multiple of d iff there exists k such that
    k*d in [L, L+S], i.e., L in [k*d - S, k*d].
    We collect all such L ranges and take their union.
    """
    # For efficiency, we'll collect candidate L values directly
    # For each divisor d, the window [L, L+S] contains a multiple of d
    # iff L <= k*d <= L+S for some k, i.e., L in [k*d - S, k*d]
    # We need L in [1, min(30030, _MAXV - S)]
    
    max_L = min(30030, _MAXV - S)
    if max_L < 1:
        return []
    
    # Collect all candidate L values
    # For each divisor, find all k such that [k*d - S, k*d] intersects [1, max_L]
    candidate_Ls = set()
    for d in _DIVISORS:
        # k*d must be such that [k*d - S, k*d] intersects [1, max_L]
        # k*d - S <= max_L and k*d >= 1
        # k >= ceil(1/d) = 1 (since d >= 6)
        # k <= floor((max_L + S) / d)
        k_min = 1
        k_max = (max_L + S) // d
        for k in range(k_min, k_max + 1):
            # Window [k*d - S, k*d] intersected with [1, max_L]
            lo = max(1, k * d - S)
            hi = min(max_L, k * d)
            if lo <= hi:
                # Add all L in [lo, hi]
                # But we only need to check L values that are "interesting"
                # Actually, we should add all L in this range
                for L in range(lo, hi + 1):
                    candidate_Ls.add(L)
    
    return sorted(candidate_Ls)


def construct():
    global _BEST, _BEST_SPAN
    start = time.time()

    # Search spans from 1 upward. For each span, check L values.
    # Due to periodicity with period 30030, we only need L in [1, 30030].
    # But the values must be <= 1000000, so L+S <= 1000000.
    
    for S in range(1, _BEST_SPAN):
        if time.time() > _DEADLINE:
            break
        
        # Generate only candidate L values where at least one divisor has a multiple
        candidate_Ls = _generate_candidate_Ls(S)
        if not candidate_Ls:
            continue
        
        found = False
        for L in candidate_Ls:
            if time.time() > _DEADLINE:
                break
            
            res = _check_window_fast(L, S)
            if res is not None:
                # Validate
                if all(1 <= x <= _MAXV for x in res) and len(set(res)) == _N:
                    if S < _BEST_SPAN:
                        _BEST = list(res)
                        _BEST_SPAN = S
                found = True
                break
        
        if found:
            break
    
    # Final validation + type coercion
    out = [int(x) for x in _BEST]
    assert len(out) == _N
    assert all(isinstance(x, int) and not isinstance(x, bool) for x in out)
    assert len(set(out)) == _N
    assert all(1 <= x <= _MAXV for x in out)
    assert all(out[i] % _DIVISORS[i] == 0 for i in range(_N))
    return out
# EVOLVE-BLOCK-END
