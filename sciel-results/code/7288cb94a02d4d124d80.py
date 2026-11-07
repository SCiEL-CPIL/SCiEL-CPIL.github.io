import time
from math import gcd
from functools import reduce

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    deadline = time.time() + 85  # 85s safety margin (90s limit)
    
    # Compute LCM of all divisors
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, divisors)  # LCM = 30030
    
    # For each divisor d, the valid multiples in [0, L) are: 0, d, 2d, ..., (L/d - 1)*d
    # So in any window [start, start+R] within one period, valid multiples for d are those
    # congruent to 0 mod d
    
    # Strategy: Search over window positions in the fundamental domain [0, L)
    # For a given window [s, s+R] (where s is in [0, L)), find if we can assign
    # distinct multiples to each divisor.
    
    # Precompute: for each divisor index i, the list of offsets (mod L) where d_i divides the value
    # Actually, simpler: for a window starting at s, the valid multiples for d_i in [s, s+R]
    # are values v where v % d_i == 0 and s <= v <= s+R
    
    # Key optimization: Since the problem is periodic with period L, we can represent
    # a window by its start position s (mod L) and size R.
    # The actual values are s + offset where offset is in [0, R].
    
    # Let me try: for R from 1 upward, for s from 0 to L-1, check if valid assignment exists.
    # L = 30030, so for each R we have 30030 windows. If R goes up to ~200, that's 6M windows.
    # Each window check needs to be very fast.
    
    # For each window [s, s+R], for each divisor d_i, the valid multiples are:
    #   first = ceil(s / d_i) * d_i
    #   multiples: first, first + d_i, first + 2*d_i, ... <= s+R
    # Count = floor((s+R)/d_i) - ceil(s/d_i) + 1 (if positive)
    
    # Use backtracking with MRV for the assignment.
    
    best_range = float('inf')
    best_assignment = None
    
    # Try increasing R values
    # For each R, try all starting positions s in [0, L)
    # But we can be smarter: only try s values that are "interesting"
    
    # Actually, let's just try all s from 0 to L-1 for each R.
    # With L=30030 and R up to maybe 500, that's 15M windows.
    # Each window: compute valid multiples (10 divisors, each has ~R/d_i multiples)
    # Backtracking: 10 variables, each with ~R/d_i options.
    
    # To make it faster, let's precompute for each s and R, the valid multiples.
    # But that's too much memory. Instead, compute on the fly.
    
    # Let me try R from 1 to 300, and for each R, s from 0 to L-1.
    # That's 300 * 30030 = 9M windows. Might be too slow.
    
    # Better: use a smarter search. Note that if a window [s, s+R] works,
    # then any window [s', s'+R] where s' ≡ s (mod L) also works (with values shifted by k*L).
    # So we only need to check s in [0, L).
    
    # Let me limit R to a reasonable range and use early termination.
    
    # Actually, the key insight: we don't need to try ALL s values.
    # For a given R, the window [s, s+R] contains valid multiples for d_i iff
    # there exists an integer k such that s <= k*d_i <= s+R.
    # This is equivalent to: floor((s+R)/d_i) > floor((s-1)/d_i) (for s > 0)
    # Or more simply: the number of multiples of d_i in [s, s+R] is at least 1.
    
    # Let me just do the brute force over s for small R, with fast pruning.
    
    def check_window(s, R):
        """Check if we can assign distinct multiples in [s, s+R] to each divisor.
        Returns the assignment or None."""
        if time.time() > deadline - 2:
            return None
        
        valid = []
        for d in divisors:
            # First multiple of d >= s
            if s == 0:
                first = 0  # But we need positive integers, so first = d
            else:
                first = ((s + d - 1) // d) * d
            # But we need positive integers, so first must be >= 1
            if first < 1:
                first = d  # Smallest positive multiple
            
            mults = []
            m = first
            while m <= s + R:
                if m >= 1:  # Positive integer
                    mults.append(m)
                m += d
            valid.append(mults)
        
        # Check if any divisor has no valid multiples
        for i, v in enumerate(valid):
            if not v:
                return None
        
        # Backtracking with MRV
        used = set()
        assignment = [None] * n
        
        def backtrack():
            if time.time() > deadline - 1:
                return False
            
            # MRV: pick the unassigned divisor with fewest available options
            min_count = float('inf')
            min_idx = -1
            for i in range(n):
                if assignment[i] is None:
                    avail = [m for m in valid[i] if m not in used]
                    if len(avail) < min_count:
                        min_count = len(avail)
                        min_idx = i
                    if min_count == 0:
                        break
            
            if min_idx == -1:
                return True  # All assigned
            if min_count == 0:
                return False
            
            for m in valid[min_idx]:
                if m not in used:
                    used.add(m)
                    assignment[min_idx] = m
                    if backtrack():
                        return True
                    used.remove(m)
                    assignment[min_idx] = None
            return False
        
        if backtrack():
            return assignment[:]
        return None
    
    # Search: try R from 1 to 500, for each R try s from 0 to L-1
    # But 500 * 30030 = 15M, too many. Let me be smarter.
    
    # Observation: For a fixed R, if window [s, s+R] works, we're done for that R.
    # So for each R, we can stop as soon as we find one valid s.
    
    # Also, we can skip s values where some divisor has no multiples in the window.
    
    for R in range(1, 500):
        if time.time() > deadline - 5:
            break
        
        found = False
        for s in range(0, L):
            if time.time() > deadline - 3:
                break
            
            # Quick check: does each divisor have at least one multiple in [s, s+R]?
            valid = True
            for d in divisors:
                # Number of multiples of d in [s, s+R]
                # floor((s+R)/d) - ceil(s/d) + 1
                if s == 0:
                    # [0, R], but we need positive, so [1, R]
                    cnt = R // d
                else:
                    first = ((s + d - 1) // d) * d
                    if first > s + R:
                        valid = False
                        break
                    cnt = (s + R - first) // d + 1
                
                if cnt < 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            result = check_window(s, R)
            if result is not None:
                # Verify
                if len(set(result)) == n and all(isinstance(x, int) and x > 0 and x <= 1000000 for x in result):
                    r = max(result) - min(result)
                    if r < best_range:
                        best_range = r
                        best_assignment = result[:]
                        if R == 1:
                            return [int(x) for x in best_assignment]
                        found = True
                        break  # Found a valid window for this R
        
        if found:
            # Found a solution with range <= R. Since R is increasing, 
            # we might find a smaller actual range with larger R? No, larger R means larger window.
            # But the actual range within the window could be smaller.
            # However, since we're going in order of R, and we found the first R with a solution,
            # any solution with smaller R would have been found earlier.
            # So this is optimal for this R, but maybe a different R gives smaller actual range?
            # No: if R' < R and there's a solution with range r' <= R', we would have found it.
            # So the best we can do is continue to see if a slightly larger R gives a smaller actual range.
            # But that's unlikely. Let's stop after a few more R values.
            # Actually, let's just stop here since we're iterating R in order.
            break
    
    if best_assignment is None:
        # Fallback
        return [int(d) for d in divisors]
    
    return [int(x) for x in best_assignment]
