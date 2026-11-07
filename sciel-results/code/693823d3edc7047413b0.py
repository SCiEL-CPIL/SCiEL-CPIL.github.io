"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000
N_DIV = len(DIVISORS)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # 1. Generate all multiples
    # all_multiples is a list of (value, divisor_index)
    all_multiples = []
    for i, d in enumerate(DIVISORS):
        for v in range(d, MAX_VAL + 1, d):
            all_multiples.append((v, i))
            
    # Sort by value
    all_multiples.sort(key=lambda x: x[0])
    
    # 2. Sliding window to find minimal span where all divisors are present
    # This gives a lower bound on the answer.
    n_total = len(all_multiples)
    counts = [0] * N_DIV
    left = 0
    min_span = float('inf')
    min_left_idx = 0
    min_right_idx = 0
    all_present = 0
    
    for right in range(n_total):
        val, div_idx = all_multiples[right]
        if counts[div_idx] == 0:
            all_present += 1
        counts[div_idx] += 1
        
        while all_present == N_DIV:
            span = val - all_multiples[left][0]
            if span < min_span:
                min_span = span
                min_left_idx = left
                min_right_idx = right
            
            l_val, l_div = all_multiples[left]
            counts[l_div] -= 1
            if counts[l_div] == 0:
                all_present -= 1
            left += 1
            
    # min_span is the minimal span where each divisor has at least one multiple.
    # The actual answer might be larger due to distinctness constraints.
    
    # 3. Search for the minimal span with distinct representatives.
    # We will check spans starting from min_span.
    # For a given span W, we check all windows of width W in the sorted list.
    
    def check_window(W, start_idx, end_idx):
        """
        Check if the subarray all_multiples[start_idx:end_idx+1] (which has width <= W)
        allows a system of distinct representatives.
        """
        # Build candidate lists for each divisor
        candidates = [[] for _ in range(N_DIV)]
        for i in range(start_idx, end_idx + 1):
            v, d_idx = all_multiples[i]
            candidates[d_idx].append(v)
            
        # Quick check: if any divisor has no candidates, fail
        for i in range(N_DIV):
            if not candidates[i]:
                return None
        
        # Bipartite Matching (Kuhn's algorithm)
        # Left: divisors (0..N_DIV-1), Right: unique values in the window
        # Note: Values in candidates might have duplicates? No, all_multiples is unique per value?
        # Wait, a value can be a multiple of multiple divisors.
        # E.g., 30 is a multiple of 6, 10, 15.
        # So in all_multiples, (30, 0), (30, 1), (30, 4) are distinct entries.
        # But for the matching, the right side is the *values*.
        # If two divisors can use the same value, that's a conflict.
        
        # Map values to indices for the right side
        all_vals = set()
        for c in candidates:
            all_vals.update(c)
        if len(all_vals) < N_DIV:
            return None
            
        val_to_idx = {v: i for i, v in enumerate(sorted(all_vals))}
        m = len(all_vals)
        
        # match_to[j] = i (left node matched to right node j)
        match_to = [-1] * m
        # match_from[i] = j (right node matched to left node i)
        match_from = [-1] * N_DIV
        
        def try_kun(u, seen):
            for v_val in candidates[u]:
                v_idx = val_to_idx[v_val]
                if seen[v_idx]:
                    continue
                seen[v_idx] = True
                if match_to[v_idx] == -1 or try_kun(match_to[v_idx], seen):
                    match_to[v_idx] = u
                    match_from[u] = v_idx
                    return True
            return False
            
        for i in range(N_DIV):
            seen = [False] * m
            if not try_kun(i, seen):
                return None
                
        # Extract values
        result = [0] * N_DIV
        idx_to_val = {i: v for v, i in val_to_idx.items()}
        for i in range(N_DIV):
            result[i] = idx_to_val[match_from[i]]
        return result

    # Iterate W from min_span upwards
    # We need to check if there is ANY window of width W that works.
    # Efficiently, we can slide a window of width W over the sorted list.
    
    best_result = None
    best_W = float('inf')
    
    # Limit W search to a reasonable bound, e.g., 5000
    # The previous code searched up to 500.
    for W in range(min_span, 5000):
        if time.time() > deadline:
            break
            
        # Slide window of width W
        # left and right pointers in all_multiples
        left = 0
        right = 0
        # We want windows where all_multiples[right][0] - all_multiples[left][0] <= W
        
        # Optimization: We only need to check windows where all divisors are present.
        # We can reuse the counts logic?
        # Simpler: Just iterate left, and find max right for this left such that width <= W.
        # Then check if that window has all divisors.
        
        # To avoid O(N^2), we can use the two-pointer approach for each W?
        # No, for each W, we can scan left from 0 to N, and right is monotonic.
        
        right = 0
        counts = [0] * N_DIV
        all_present = 0
        
        for left in range(n_total):
            # Ensure right >= left
            if right < left:
                right = left
            
            # Expand right as far as possible within width W
            while right < n_total and all_multiples[right][0] - all_multiples[left][0] <= W:
                v, d_idx = all_multiples[right]
                if counts[d_idx] == 0:
                    all_present += 1
                counts[d_idx] += 1
                right += 1
                
            # Now window [left, right-1] is the maximal window starting at left with width <= W
            if all_present == N_DIV:
                # Check for distinct representatives
                res = check_window(W, left, right - 1)
                if res:
                    # Found a solution with span <= W
                    # Since we iterate W upwards, this is the minimal W for this specific window?
                    # Not necessarily, but we return the first valid one found.
                    # To be safe, we can break and return.
                    return [int(v) for v in res]
            
            # Shrink left
            l_val, l_div = all_multiples[left]
            counts[l_div] -= 1
            if counts[l_div] == 0:
                all_present -= 1
                
    # Fallback if no solution found
    # This should not happen for W up to 5000
    # Return divisors themselves as valid (though span might be large)
    return [int(d) for d in DIVISORS]

# EVOLVE-BLOCK-END
