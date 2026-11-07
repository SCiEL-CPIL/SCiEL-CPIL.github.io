import time
from itertools import product

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    deadline = time.time() + 80  # 80s safety margin
    
    best_range = float('inf')
    best_assignment = None
    
    # For each divisor, we need to pick a multiple.
    # Strategy: try increasing range sizes R.
    # For each R, try various window positions.
    
    # Precompute: for a given window [L, L+R], for each divisor d,
    # the valid multiples are L + offset where (L+offset) % d == 0.
    
    # Use backtracking with MRV heuristic.
    
    def check_window(L, R, divisors):
        """Check if we can assign distinct multiples in [L, L+R]."""
        n = len(divisors)
        # For each divisor, find valid multiples in [L, L+R]
        valid = []
        for d in divisors:
            # First multiple >= L
            first = ((L + d - 1) // d) * d
            # Generate all multiples in [L, L+R]
            mults = []
            m = first
            while m <= L + R:
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
        
        def backtrack(idx):
            if idx == n:
                return True
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
                return True
            if min_count == 0:
                return False
            
            for m in valid[min_idx]:
                if m not in used:
                    used.add(m)
                    assignment[min_idx] = m
                    if backtrack(idx + 1):
                        return True
                    used.remove(m)
                    assignment[min_idx] = None
            return False
        
        if backtrack(0):
            return assignment[:]
        return None
    
    # Try increasing R values
    # Start with small R and increase
    # For each R, try a range of L values
    
    # Heuristic: L should be such that the window contains good multiples.
    # Try L from 0 to some max, stepping by 1 or by gcd.
    
    # Actually, let's try R from 1 to 500, and for each R, try L from 0 to 5000.
    # That's 500 * 5000 = 2.5M windows, each with backtracking. Too slow.
    
    # Better: for each R, try a limited set of L values.
    # Or: use a smarter search.
    
    # Alternative: random restarts with local search.
    
    # Let me try: for R in range(1, 200):
    #   for L in range(0, 10000, step):
    #     try to find assignment
    
    # But this is still potentially slow. Let me limit it.
    
    # Actually, let me try a different approach: 
    # For each divisor, the multiples are dense enough that in a window of size ~100,
    # each divisor has several options. So backtracking should be fast.
    
    # Let's try R from 1 to 300, and for each R, try L from 0 to 2000 step 1.
    # That's 300 * 2000 = 600K windows. Each backtracking is fast (10 vars).
    # Should be doable in < 60s.
    
    for R in range(1, 300):
        if time.time() > deadline - 10:
            break
        # Try various L positions
        # To be efficient, try L in a reasonable range
        # The values are at most 1000000, so L can be up to 1000000 - R
        # But small values are fine too.
        
        # Try L from 0 to 5000, step 1
        for L in range(0, 5001):
            if time.time() > deadline - 5:
                break
            result = check_window(L, R, divisors)
            if result is not None:
                # Verify distinctness
                if len(set(result)) == n:
                    r = max(result) - min(result)
                    if r < best_range:
                        best_range = r
                        best_assignment = result[:]
                        # Found a valid solution with this R, can break
                        # But maybe smaller R exists? No, we're going in order.
                        # Actually R here is the window size, but the actual range
                        # might be smaller. Let's record it.
                        if R == 1:  # Can't do better than 1 (distinct positive ints)
                            return [int(x) for x in best_assignment]
                        # Continue searching for potentially better (smaller actual range)
                        # But since we're increasing R, and we found one with this R,
                        # any future R will be larger. But the actual range within the window
                        # might be smaller than R.
                        # Let's break out of L loop and try next R? No, we want to
                        # find the minimum actual range.
                        # Actually, since we're iterating R from 1 up, the first R for
                        # which we find ANY valid window gives us a solution with range <= R.
                        # But the actual range might be < R. We should record it and
                        # continue to see if a smaller actual range is possible with same or larger R.
                        # But since R is increasing, we can't get a smaller window.
                        # However, the actual range within a window of size R could be anything
                        # from 1 to R. So we should keep searching.
                        # Let's just record and continue.
                        pass
        # End L loop
        # If we found a solution with this R, we can stop if R is small enough
        # Actually, let's continue to R+1 only if we haven't found a solution yet,
        # or if we want to check if a smaller actual range is possible.
        # Since we're going in order of R, the first R with a solution gives range <= R.
        # We should continue a bit to see if we can find a smaller actual range.
        # But for efficiency, let's stop after finding a solution and a few more R values.
        if best_assignment is not None:
            # Found a solution. Let's try a few more R values to see if we can
            # find a smaller actual range.
            # But since R is increasing, the window is larger, so it's unlikely to
            # find a smaller range. Let's stop.
            break
    
    if best_assignment is None:
        # Fallback: return the divisors themselves (valid but not optimal)
        return [int(d) for d in divisors]
    
    return [int(x) for x in best_assignment]
