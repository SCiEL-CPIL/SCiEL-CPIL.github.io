"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import permutations

P = {'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39], 'maximum': 1000000}
DIVISORS = P['divisors']
MAX_VAL = P['maximum']
N = len(DIVISORS)

def _find_multiple_in_range(d, lo, hi):
    """Find the smallest multiple of d in [lo, hi], or None if no such multiple exists."""
    if hi < lo:
        return None
    m = ((lo + d - 1) // d) * d
    if m > hi:
        return None
    return m

def _check_window(L, W, divisors):
    """
    Check if there exist distinct multiples of each divisor in [L, L+W].
    Returns the assignment (list of values) if valid, else None.
    Uses backtracking with ordering by divisor size (larger divisors have fewer choices).
    """
    hi = L + W
    if hi > MAX_VAL:
        return None
    
    # Sort indices by divisor size descending (larger divisors have fewer multiples in window)
    # Actually, smaller divisors have MORE multiples, so they are more flexible.
    # Larger divisors have FEWER multiples, so they are more constrained.
    # We should assign the most constrained (largest divisors) first?
    # Or assign the ones with fewest options first.
    # Let's calculate the number of multiples for each divisor in [L, L+W].
    
    options = []
    for i, d in enumerate(divisors):
        count = 0
        first = ((L + d - 1) // d) * d
        if first <= hi:
            count = (hi - first) // d + 1
        options.append((count, i, d))
    
    # Sort by count ascending (most constrained first)
    options.sort(key=lambda x: x[0])
    
    # If any has 0 options, return None
    if options[0][0] == 0:
        return None
    
    assigned = [None] * N
    used = set()
    
    def backtrack(idx):
        if idx == N:
            return True
        
        count, orig_idx, d = options[idx]
        lo = L
        hi = L + W
        
        # Generate all multiples of d in [lo, hi]
        first = ((lo + d - 1) // d) * d
        if first > hi:
            return False
            
        for m in range(first, hi + 1, d):
            if m not in used:
                used.add(m)
                assigned[orig_idx] = m
                if backtrack(idx + 1):
                    return True
                used.remove(m)
                assigned[orig_idx] = None
        
        return False
    
    if backtrack(0):
        return assigned
    else:
        return None

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Strategy:
    # 1. Try small windows W starting from 1 upwards.
    # 2. For each W, try different starting points L.
    # 3. L should be such that [L, L+W] contains potential multiples.
    #    A good heuristic is to try L values that are close to multiples of the LCM or divisors.
    #    Or simply scan L from 1 to MAX_VAL - W, but that's too slow.
    
    # Better approach:
    # The minimum possible span is bounded by the density of multiples.
    # For a window [L, L+W], the number of multiples of d is roughly W/d.
    # We need 10 distinct numbers.
    
    # Let's try W from 1 to some limit.
    # For a given W, we can try L values that are "aligned" with the divisors.
    # Specifically, L can be any integer. But we can restrict L to be a multiple of some gcd or similar?
    # No, L can be anything.
    
    # Heuristic:
    # Start with W=1. Check if any window of size 1 works (impossible for 10 distinct).
    # Increase W.
    # For each W, try a set of candidate L values.
    # Candidate L values:
    # - L such that L is a multiple of each divisor? No, L is just the start.
    # - We can try L = k * lcm(divisors) - offset?
    # - Simpler: Try L from 1 to some limit, but step by 1? Too slow.
    
    # Let's use a randomized or structured search for L.
    # For a fixed W, the probability that a random window [L, L+W] works is:
    # Product over i of (W/d_i + 1) / (W+1) roughly?
    # If this probability is high, few trials needed. If low, many.
    
    # Let's estimate the "density".
    # For W=100, d=6 has ~16 multiples, d=10 has ~10, etc.
    # The constraints are tight.
    
    # Alternative: Use the fact that the numbers must be distinct.
    # Let's try to find a window where the "slots" are well-filled.
    
    # Algorithm:
    # 1. Iterate W from 1 to 1000 (or until deadline).
    # 2. For each W, try a limited number of L values.
    #    Which L values?
    #    We can try L such that L mod d_i is "nice"?
    #    Actually, let's just try L = 1, 2, ..., up to some limit, but that's slow.
    
    # Smarter:
    # The optimal L is likely to be small? Or distributed?
    # Let's try L values that are multiples of the smallest divisor (6)?
    # Or L values that make the window centered on multiples?
    
    # Let's try a different approach:
    # Pick a target value T. Try to find multiples near T.
    # This is equivalent to scanning windows.
    
    # Let's scan W from 1 to 200.
    # For each W, try L from 1 to MAX_VAL, but step by something?
    # No, we need to be careful.
    
    # Let's try specific L values:
    # L = 1, 6, 10, 14, ... (multiples of divisors)
    # And L = 1, 2, 3, ... (small L)
    
    # Actually, let's just iterate L from 1 to 10000 for each W?
    # 10000 * 200 = 2,000,000 checks.
    # Each check involves backtracking.
    # Backtracking might be fast if it fails quickly.
    
    # Let's try:
    best_result = None
    best_span = MAX_VAL
    
    # Pre-compute some useful info
    # For a given W, what is the max number of multiples?
    
    for W in range(1, 500):
        if time.time() > deadline:
            break
            
        # Try a set of L values
        # 1. Small L: 1 to 100
        # 2. L aligned with divisors: multiples of 6, 10, etc.
        # Let's try L from 1 to 1000, step 1.
        # And also L = k * 600 (lcm of 6,10,14... no, lcm is large)
        
        # Let's try L = 1 to 2000.
        found = False
        for L in range(1, 2000):
            if time.time() > deadline:
                break
                
            result = _check_window(L, W, DIVISORS)
            if result is not None:
                # Verify distinctness and constraints
                if len(set(result)) == N:
                    if all(0 < result[i] <= MAX_VAL for i in range(N)):
                        if all(result[i] % DIVISORS[i] == 0 for i in range(N)):
                            span = W
                            if span < best_span:
                                best_span = span
                                best_result = result[:]
                            found = True
                            break # Found one for this W, can stop searching L for this W?
                            # Wait, we want the MINIMUM span. If we found one for W,
                            # we can break the W loop entirely?
                            # No, we are iterating W from 1 up. So the first W we find
                            # is the minimum W?
                            # Yes! Because we start from W=1 and increase.
                            # So the first valid W is the optimal W.
                            # But we need to ensure we found the GLOBAL minimum.
                            # Since we scan W in increasing order, the first success is optimal.
                            return [int(x) for x in best_result]
        
        # If we didn't find any L in 1..2000 for this W, try more L?
        # Or move to next W?
        # If W is small, the valid L might be sparse.
        # Let's try a wider range of L if W is small.
        
        if W <= 20:
            # Try L from 2000 to 10000
            for L in range(2000, 10000, 10): # Step 10 to save time
                if time.time() > deadline:
                    break
                result = _check_window(L, W, DIVISORS)
                if result is not None:
                    if len(set(result)) == N:
                        if all(0 < result[i] <= MAX_VAL for i in range(N)):
                            if all(result[i] % DIVISORS[i] == 0 for i in range(N)):
                                return [int(x) for x in result]
    
    # If we exhausted the search, return the best found or a fallback
    if best_result:
        return [int(x) for x in best_result]
    
    # Fallback: Just return the divisors themselves (which is valid but span is large)
    return [int(d) for d in DIVISORS]

# EVOLVE-BLOCK-END
