import time
from math import gcd
from functools import reduce

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
N = len(DIVISORS)
MAX_VAL = 1000000
PERIOD = reduce(lambda a, b: a * b // gcd(a, b), DIVISORS)  # 60060

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = N
    divisors = DIVISORS
    max_val = MAX_VAL
    period = PERIOD

    # Precompute: for each divisor, how many multiples in [1, max_val]
    # and their positions in a global sorted list approach
    
    # Strategy: Sliding window over the sorted list of all valid multiples.
    # Each number v in [1, max_val] is "covered" by divisor i if v % divisors[i] == 0.
    # We want the shortest interval [a, b] such that for every divisor i, 
    # there exists some multiple of divisors[i] in [a, b].
    
    # Key insight: The set of all multiples of any divisor in [1, max_val] can be 
    # represented as a sorted list. We use a two-pointer/sliding window approach.
    # For each position in the sorted list of "events" (multiples), we track which 
    # divisors are still "satisfied" within the current window.
    
    # However, generating all multiples up to 1,000,000 for 10 divisors is feasible:
    # Sum of max_val / divisors[i] ≈ 1000000 * (1/6 + 1/10 + 1/14 + ...) ≈ 1000000 * 0.5 ≈ 500,000 entries
    
    # Let's build a list of (value, divisor_index) for all multiples, sort by value,
    # then use sliding window to find the minimum span where all 10 divisors are covered.
    
    # Actually, we need to be careful: we want distinct values, one per divisor.
    # The sliding window approach: 
    # - Create events: for each divisor i, for each multiple m of divisors[i] in [1, max_val],
    #   add event (m, i)
    # - Sort all events by m
    # - Use two pointers (left, right) to maintain a window where all 10 divisors appear at least once
    # - Track the minimum (right_value - left_value) such that the window [left_value, right_value]
    #   contains at least one multiple for each divisor
    
    # But we need DISTINCT values. If two divisors share the same multiple value, that's fine 
    # as long as we pick different values. The sliding window finds the minimal span where 
    # each divisor has at least one candidate. Then we need to verify we can assign distinct values.
    
    # Let's implement the sliding window:
    
    # Build events list
    events = []
    for i in range(n):
        d = divisors[i]
        m = d
        while m <= max_val:
            events.append((m, i))
            m += d
    
    events.sort()  # Sort by value
    
    # Sliding window: find minimum span where all n divisors are represented
    # count[i] = number of multiples of divisor i in current window
    best_diff = float('inf')
    best_left_val = -1
    best_right_val = -1
    
    count = [0] * n
    covered = 0  # Number of divisors with count > 0
    
    left = 0
    for right in range(len(events)):
        val_r, div_r = events[right]
        count[div_r] += 1
        if count[div_r] == 1:
            covered += 1
        
        while covered == n and left <= right:
            val_l, div_l = events[left]
            span = val_r - val_l
            if span < best_diff:
                best_diff = span
                best_left_val = val_l
                best_right_val = val_r
            
            count[div_l] -= 1
            if count[div_l] == 0:
                covered -= 1
            left += 1
    
    # Now we have the best window [best_left_val, best_right_val].
    # We need to find an actual assignment of distinct values within this window.
    # Use backtracking with MRV.
    
    def find_assignment(L, R):
        """Find distinct multiples for each divisor in [L, R]."""
        W = R - L + 1
        if W < n:
            return None
        
        cands = []
        for i in range(n):
            d = divisors[i]
            # Find all multiples of d in [L, R]
            first = ((L + d - 1) // d) * d  # Smallest multiple of d >= L
            c = []
            m = first
            while m <= R:
                c.append(m)
                m += d
            if not c:
                return None
            cands.append(c)
        
        # MRV: sort by number of candidates
        order = sorted(range(n), key=lambda i: len(cands[i]))
        
        used = set()
        assignment = [0] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            i = order[idx]
            for v in cands[i]:
                if v not in used:
                    used.add(v)
                    assignment[i] = v
                    if backtrack(idx + 1):
                        return True
                    used.remove(v)
            return False
        
        if backtrack(0):
            result = [assignment[i] for i in range(n)]
            # Validate
            if len(set(result)) == n:
                for i in range(n):
                    if not (isinstance(result[i], int) and result[i] % divisors[i] == 0 and 1 <= result[i] <= max_val):
                        return None
                return result
        return None
    
    if best_left_val != -1:
        result = find_assignment(best_left_val, best_right_val)
        if result:
            return result
    
    # Fallback: try a few nearby windows or use a simpler approach
    # Try expanding the window slightly
    for expansion in range(1, 50):
        L = max(1, best_left_val - expansion)
        R = min(max_val, best_right_val + expansion)
        result = find_assignment(L, R)
        if result:
            return result
    
    # Final fallback: use the divisors themselves (they are distinct and valid)
    return [divisors[i] for i in range(n)]
