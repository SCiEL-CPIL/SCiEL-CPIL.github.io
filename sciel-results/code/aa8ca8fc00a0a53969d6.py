import time
from math import gcd
from functools import reduce

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin (90s limit)
    
    # Compute LCM of all divisors
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, divisors)  # LCM = 30030
    
    # For a given window [s, s+R], we need to assign distinct multiples of each divisor
    # within that window. The key insight is that the problem is periodic with period L.
    
    # Strategy: For each R from 1 upward, check if there exists a window [s, s+R] (mod L)
    # where we can assign distinct multiples. We use backtracking with MRV heuristic.
    
    best_range = float('inf')
    best_assignment = None
    
    def check_window(s, R):
        """Check if we can assign distinct multiples in [s, s+R] to each divisor.
        Returns the assignment or None."""
        if time.time() > deadline - 2:
            return None
        
        # Compute valid multiples for each divisor in [s, s+R]
        valid = []
        for d in divisors:
            # First multiple of d >= s (and >= 1)
            if s <= 0:
                first = d  # Smallest positive multiple
            else:
                first = ((s + d - 1) // d) * d
                if first < 1:
                    first = d
            
            mults = []
            m = first
            while m <= s + R:
                if m >= 1:
                    mults.append(m)
                m += d
            valid.append(mults)
        
        # Check if any divisor has no valid multiples
        for v in valid:
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
            
            # Try values in order of least constraining value heuristic (just use natural order)
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
    
    # Quick check: does each divisor have at least one multiple in [s, s+R]?
    def has_multiple(s, R, d):
        if s <= 0:
            # Window [s, s+R] includes 0 or starts before 1
            # We need positive multiples, so check [max(1,s), s+R]
            low = max(1, s)
            if low > s + R:
                return False
            first = ((low + d - 1) // d) * d
            return first <= s + R
        else:
            first = ((s + d - 1) // d) * d
            return first <= s + R
    
    # Precompute for each divisor d, the residues r in [0, L-1] such that
    # there exists a multiple of d in [r, r+R] (for a given R)
    # This is equivalent to: there exists k such that r <= k*d <= r+R
    # i.e., k*d mod L is in [r, r+R] mod L
    
    # For efficiency, for each R, we can precompute which s values work
    # But let's first try a smarter approach: for each R, only check s values
    # that are "interesting" - specifically, s values where at least one
    # multiple boundary is crossed.
    
    # Actually, let's use a different approach: for each R, the set of valid s
    # values is a union of intervals. We can find these intervals.
    
    # For divisor d, the set of s in [0, L-1] such that there's a multiple of d
    # in [s, s+R] is:
    # For each multiple m = k*d, the s values that work are those where
    # s <= m <= s+R, i.e., m-R <= s <= m
    # But s is in [0, L-1], and we also need s+R >= 1 (for positive multiples)
    
    # Let's compute the union of intervals [m-R, m] for all multiples m of d
    # that fall in [1, L+R] (since s is in [0, L-1] and s+R can go up to L-1+R)
    
    def compute_valid_s_intervals(R, d):
        """Compute the set of s in [0, L-1] such that there's a positive multiple
        of d in [s, s+R]. Returns a list of (start, end) intervals."""
        intervals = []
        # Multiples of d in [1, L+R]
        first_mult = d
        last_mult = ((L + R) // d) * d
        if last_mult > L + R:
            last_mult -= d
        
        m = first_mult
        while m <= last_mult:
            # s must satisfy: s <= m <= s+R, and 0 <= s <= L-1
            # So: max(0, m-R) <= s <= min(L-1, m)
            s_low = max(0, m - R)
            s_high = min(L - 1, m)
            if s_low <= s_high:
                intervals.append((s_low, s_high))
            m += d
        
        # Merge overlapping intervals
        if not intervals:
            return []
        
        intervals.sort()
        merged = [intervals[0]]
        for start, end in intervals[1:]:
            if start <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        
        return merged
    
    # Search: try R from 1 to 500
    for R in range(1, 500):
        if time.time() > deadline - 5:
            break
        
        # Compute valid s intervals for each divisor
        valid_intervals = []
        all_valid = True
        for d in divisors:
            intervals = compute_valid_s_intervals(R, d)
            if not intervals:
                all_valid = False
                break
            valid_intervals.append(intervals)
        
        if not all_valid:
            continue
        
        # Find the intersection of all valid s sets
        # Start with the first divisor's intervals
        current_intervals = valid_intervals[0]
        
        for i in range(1, n):
            new_intervals = []
            for s1_start, s1_end in current_intervals:
                for s2_start, s2_end in valid_intervals[i]:
                    # Intersection of [s1_start, s1_end] and [s2_start, s2_end]
                    inter_start = max(s1_start, s2_start)
                    inter_end = min(s1_end, s2_end)
                    if inter_start <= inter_end:
                        new_intervals.append((inter_start, inter_end))
            current_intervals = new_intervals
            if not current_intervals:
                break
        
        if not current_intervals:
            continue
        
        # For each interval of valid s values, try to find a solution
        found = False
        for s_start, s_end in current_intervals:
            if time.time() > deadline - 3:
                break
            
            # Try s values in this interval
            # To be efficient, we can try a few sample points
            # or just iterate through all (if the interval is small)
            
            # For now, let's try all s values in the interval
            for s in range(s_start, s_end + 1):
                if time.time() > deadline - 3:
                    break
                
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
                            break
            
            if found:
                break
        
        if found:
            break
    
    if best_assignment is None:
        # Fallback: use the divisors themselves
        return [int(d) for d in divisors]
    
    return [int(x) for x in best_assignment]
