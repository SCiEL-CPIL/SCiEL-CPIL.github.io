"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import product

_DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
_LCM = 30030
_MAX_VAL = 1000000
_N = 10

def _precompute_bitsets():
    """Precompute bitset of positions (0..LCM-1) where each divisor has a multiple."""
    bitsets = []
    for d in _DIVISORS:
        bs = 0
        for pos in range(0, _LCM, d):
            bs |= (1 << pos)
        bitsets.append(bs)
    return bitsets

def _check_window_offset(offset, R, bitsets):
    """Check if window [offset, offset+R] (mod LCM) has distinct multiples for all divisors.
    Returns assignment (list of positions) if feasible, else None."""
    # For each divisor, the available positions in the window are those in [offset, offset+R] mod LCM
    # We need to pick one position per divisor such that all are distinct.
    # This is a bipartite matching / exact cover problem.
    # Since N=10 is small, use backtracking.
    
    # Get available positions for each divisor
    avail = []
    for i, d in enumerate(_DIVISORS):
        bs = bitsets[i]
        # Positions in [offset, offset+R] that are multiples of d (mod LCM)
        # A position p (0..LCM-1) is in window if (p - offset) % LCM <= R
        # i.e., p is in [offset, offset+R] mod LCM
        positions = []
        # Check each multiple of d in [0, LCM)
        for p in range(0, _LCM, d):
            if (p - offset) % _LCM <= R:
                positions.append(p)
        avail.append(positions)
    
    # Backtracking: assign distinct positions to each divisor
    used = set()
    assignment = [None] * _N
    
    def backtrack(idx):
        if idx == _N:
            return True
        for p in avail[idx]:
            if p not in used:
                used.add(p)
                assignment[idx] = p
                if backtrack(idx + 1):
                    return True
                used.remove(p)
                assignment[idx] = None
        return False
    
    if backtrack(0):
        return assignment
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    bitsets = _precompute_bitsets()
    
    best_R = None
    best_assignment = None
    best_offset = None
    
    # Binary search on R (window size)
    # Lower bound: at least 1 (distinct integers)
    # Upper bound: LCM (30030) is always feasible
    lo = 1
    hi = _LCM
    
    # First, verify hi is feasible
    # Try all offsets for a given R
    def is_feasible(R):
        if time.time() > deadline:
            return None  # Timeout, use current best
        
        for offset in range(_LCM):
            if time.time() > deadline:
                return None
            result = _check_window_offset(offset, R, bitsets)
            if result is not None:
                return (offset, result)
        return None
    
    # Binary search for minimum R
    while lo < hi:
        if time.time() > deadline:
            break
        mid = (lo + hi) // 2
        res = is_feasible(mid)
        if res is None:
            # Timeout or infeasible
            if time.time() > deadline:
                break
            lo = mid + 1
        else:
            offset, assignment = res
            best_R = mid
            best_assignment = assignment
            best_offset = offset
            hi = mid
    
    # If binary search didn't find anything (timeout early), try a direct approach
    if best_R is None:
        # Try increasing R from a reasonable start
        for R in range(10, _LCM + 1):
            if time.time() > deadline:
                break
            for offset in range(_LCM):
                if time.time() > deadline:
                    break
                result = _check_window_offset(offset, R, bitsets)
                if result is not None:
                    best_R = R
                    best_assignment = result
                    best_offset = offset
                    break
            if best_R is not None:
                break
    
    if best_R is None:
        # Fallback: just return the divisors themselves (valid but not optimal)
        return list(_DIVISORS)
    
    # Now construct actual integers from the assignment
    # The assignment gives positions p_i in [0, LCM) for each divisor i.
    # We need to find a base S such that S + p_i is a multiple of divisor_i for all i,
    # and all S + p_i are distinct positive integers <= 1000000.
    # 
    # S + p_i ≡ 0 (mod d_i) means S ≡ -p_i (mod d_i)
    # We need S such that S ≡ -p_i (mod d_i) for all i.
    # This is a system of congruences. Since LCM(d_i) = LCM, and p_i are chosen from
    # multiples of d_i mod LCM, we need to check consistency.
    # 
    # Actually, let's think differently. The position p_i means that in the window
    # starting at offset, the multiple is at position p_i. The actual value is:
    # S + p_i where S is a multiple of LCM (so the pattern repeats).
    # Wait, no. Let me reconsider.
    #
    # If offset is the start of the window mod LCM, then the window is [offset, offset+R] mod LCM.
    # A position p in this window corresponds to values ... , offset + p - offset, ... 
    # Actually, the values in the window are: for k=0,1,2,..., the value offset + k (mod LCM)
    # corresponds to actual values S + offset + k where S is a multiple of LCM.
    #
    # So if position p_i is selected for divisor i, the actual value is S + p_i where
    # S ≡ 0 (mod LCM) is a multiple of LCM. But we need S + p_i to be divisible by d_i.
    # Since p_i is a multiple of d_i (mod LCM), and LCM is a multiple of d_i,
    # S + p_i ≡ 0 + p_i ≡ p_i ≡ 0 (mod d_i). Good.
    #
    # So the actual values are S + p_i for i=0..9, where S is a multiple of LCM.
    # We need:
    # 1. All S + p_i are distinct: since p_i are distinct (from backtracking), this is satisfied.
    # 2. All S + p_i >= 1: S >= 1 - min(p_i). Since S is a multiple of LCM >= 0,
    #    and p_i >= 0, we need S + min(p_i) >= 1.
    # 3. All S + p_i <= 1000000: S + max(p_i) <= 1000000, so S <= 1000000 - max(p_i).
    #
    # To minimize max - min = max(p_i) - min(p_i) (since S is common), we just need
    # to find a valid S. The range is independent of S.
    #
    # Let's find S:
    # S must be a multiple of LCM, S >= max(0, 1 - min_p), S <= 1000000 - max_p
    min_p = min(best_assignment)
    max_p = max(best_assignment)
    S_min = max(0, 1 - min_p)
    S_max = 1000000 - max_p
    
    # Find a multiple of LCM in [S_min, S_max]
    S = None
    if S_min <= S_max:
        # Smallest multiple of LCM >= S_min
        S = ((S_min + _LCM - 1) // _LCM) * _LCM
        if S > S_max:
            S = None
    
    if S is None:
        # Try to adjust: maybe we can shift the window
        # For now, fallback
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
