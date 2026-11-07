"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

_DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
_LCM = 30030
_MAX_VAL = 1000000
_N = 10

def _get_positions(d, offset, R):
    """Get positions p in [0, LCM) that are multiples of d and in window [offset, offset+R] mod LCM.
    Uses direct arithmetic instead of iterating all multiples."""
    if R >= _LCM - 1:
        # Entire circle is covered
        return list(range(0, _LCM, d))
    # Window in modular space: [offset, offset+R] mod LCM
    # A position p is in window if (p - offset) % LCM <= R
    # This means p is in [offset, offset+R] (if offset+R < LCM)
    # or p is in [offset, LCM-1] U [0, (offset+R)%LCM] (if offset+R >= LCM)
    end = offset + R
    if end < _LCM:
        # Single interval [offset, end]
        # Find multiples of d in [offset, end]
        first = ((offset + d - 1) // d) * d
        if first > end:
            return []
        last = (end // d) * d
        return list(range(first, last + 1, d))
    else:
        # Two intervals: [offset, LCM-1] and [0, end % LCM]
        positions = []
        # First part: [offset, LCM-1]
        first1 = ((offset + d - 1) // d) * d
        if first1 < _LCM:
            positions.extend(range(first1, _LCM, d))
        # Second part: [0, end % LCM]
        end2 = end % _LCM
        last2 = (end2 // d) * d
        if last2 >= 0:
            positions.extend(range(0, last2 + 1, d))
        return positions

def _check_window_offset(offset, R):
    """Check if window [offset, offset+R] (mod LCM) has distinct multiples for all divisors.
    Returns assignment (list of positions) if feasible, else None."""
    avail = []
    for i, d in enumerate(_DIVISORS):
        positions = _get_positions(d, offset, R)
        if not positions:
            return None
        avail.append(positions)
    
    # Sort by number of available positions (most constrained first) for better pruning
    order = sorted(range(_N), key=lambda i: len(avail[i]))
    
    used = set()
    assignment = [None] * _N
    
    def backtrack(idx):
        if idx == _N:
            return True
        i = order[idx]
        for p in avail[i]:
            if p not in used:
                used.add(p)
                assignment[i] = p
                if backtrack(idx + 1):
                    return True
                used.remove(p)
                assignment[i] = None
        return False
    
    if backtrack(0):
        return assignment
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin (90s limit)
    
    best_R = None
    best_assignment = None
    best_offset = None
    
    def is_feasible(R):
        if time.time() > deadline:
            return None
        
        for offset in range(_LCM):
            if time.time() > deadline:
                return None
            result = _check_window_offset(offset, R)
            if result is not None:
                return (offset, result)
        return None
    
    # Binary search for minimum R
    lo = 1
    hi = _LCM - 1  # R=LCM-1 covers entire circle, always feasible
    
    while lo < hi:
        if time.time() > deadline:
            break
        mid = (lo + hi) // 2
        res = is_feasible(mid)
        if res is None:
            lo = mid + 1
        else:
            offset, assignment = res
            best_R = mid
            best_assignment = assignment
            best_offset = offset
            hi = mid
    
    if best_R is None:
        # Fallback: just return the divisors themselves (valid but not optimal)
        return list(_DIVISORS)
    
    # Now construct actual integers from the assignment
    min_p = min(best_assignment)
    max_p = max(best_assignment)
    span_in_window = max_p - min_p  # This is the actual span of positions
    
    # We need S such that:
    #   S + min_p >= 1  =>  S >= 1 - min_p
    #   S + max_p <= 1000000  =>  S <= 1000000 - max_p
    # And S must be a multiple of LCM (so that S + p is divisible by d_i)
    # Wait - actually S doesn't need to be a multiple of LCM.
    # We need (S + p_i) % d_i == 0 for each i.
    # Since p_i % d_i == 0 (p_i is a multiple of d_i), we need S % d_i == 0 for all i.
    # So S must be a multiple of LCM.
    
    S_min = max(0, 1 - min_p)
    S_max = 1000000 - max_p
    
    if S_min > S_max:
        return list(_DIVISORS)
    
    # Find the S (multiple of LCM) in [S_min, S_max] that minimizes the actual span
    # The actual span is always max_p - min_p regardless of S (since all values shift by S)
    # So any valid S gives the same span. Just pick the smallest valid S.
    S = ((S_min + _LCM - 1) // _LCM) * _LCM
    if S > S_max:
        # Try next lower multiple? No, we need S >= S_min.
        # If no multiple of LCM exists in [S_min, S_max], we have a problem.
        # But since LCM=30030 and the range is at most 1000000, there should be one.
        return list(_DIVISORS)
    
    result = [S + p for p in best_assignment]
    
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
