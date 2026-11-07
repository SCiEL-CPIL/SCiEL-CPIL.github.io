"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def _all_lcm(divs):
    result = 1
    for d in divs:
        result = _lcm(result, d)
    return result

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000
N = len(DIVISORS)

def _get_candidates(L, W, div_idx):
    """Get all multiples of DIVISORS[div_idx] in [L, L+W]."""
    d = DIVISORS[div_idx]
    # First multiple >= L
    k_start = (L + d - 1) // d
    # Last multiple <= L+W
    k_end = (L + W) // d
    if k_start > k_end:
        return []
    return [k * d for k in range(k_start, k_end + 1)]

def _check_window(L, W):
    """Check if there exist distinct multiples for all divisors in [L, L+W]."""
    if L < 1:
        return None
    if L + W > MAX_VAL:
        return None
    
    # Get candidates for each divisor
    candidates = []
    for i in range(N):
        cands = _get_candidates(L, W, i)
        if not cands:
            return None
        candidates.append(cands)
    
    # Backtracking: assign a value to each divisor, all distinct
    assigned = [None] * N
    used = set()
    
    def backtrack(idx):
        if idx == N:
            return True
        for val in candidates[idx]:
            if val not in used:
                used.add(val)
                assigned[idx] = val
                if backtrack(idx + 1):
                    return True
                used.remove(val)
                assigned[idx] = None
        return False
    
    if backtrack(0):
        return assigned[:]
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s
    
    # Binary search on W (span)
    # Lower bound: 9 (since we need 10 distinct integers, min span is at least 9)
    # Upper bound: something reasonable
    lo = 9
    hi = 1000  # Start with a reasonable upper bound
    
    best_result = None
    best_W = None
    
    # First, find a feasible W with binary search
    # Check if hi is feasible
    found_hi = False
    for L in range(1, min(MAX_VAL - hi + 1, 100000) + 1):
        if time.time() > deadline:
            break
        result = _check_window(L, hi)
        if result is not None:
            best_result = result
            best_W = hi
            found_hi = True
            break
    
    if not found_hi:
        # Expand hi
        hi = 5000
        for L in range(1, min(MAX_VAL - hi + 1, 100000) + 1):
            if time.time() > deadline:
                break
            result = _check_window(L, hi)
            if result is not None:
                best_result = result
                best_W = hi
                found_hi = True
                break
    
    if not found_hi:
        # Fallback: just use the divisors themselves if they work
        # Check if divisors are all distinct and <= MAX_VAL
        divs = DIVISORS[:]
        if len(set(divs)) == N and max(divs) <= MAX_VAL:
            return divs
        # Return something valid
        return [d * 1 for d in DIVISORS]
    
    # Binary search for minimum W
    while lo < hi and (time.time() < deadline - 5):
        mid = (lo + hi) // 2
        found_mid = False
        for L in range(1, min(MAX_VAL - mid + 1, 100000) + 1):
            if time.time() > deadline - 5:
                break
            result = _check_window(L, mid)
            if result is not None:
                best_result = result
                best_W = mid
                found_mid = True
                break
        if found_mid:
            hi = mid
        else:
            lo = mid + 1
    
    # Now search for the best (smallest span) solution with the current best_W
    # Actually, let's do a more thorough search: try all W from best_W downward isn't needed
    # since we binary searched. But let's verify and try to find even better.
    
    # Final: return the best result found
    if best_result is not None:
        # Verify it's valid
        assert len(best_result) == N
        assert all(isinstance(x, int) for x in best_result)
        assert len(set(best_result)) == N
        assert all(x >= 1 and x <= MAX_VAL for x in best_result)
        for i in range(N):
            assert best_result[i] % DIVISORS[i] == 0
        return list(best_result)
    
    # Fallback
    return DIVISORS[:]

# EVOLVE-BLOCK-END
