import time
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    # LCM of all divisors
    LCM = 1
    for d in divisors:
        LCM = _lcm(LCM, d)
    # LCM = 30030

    start_time = time.time()
    deadline = start_time + 80.0

    best_span = float('inf')
    best_reps = [0] * n

    # Sort divisors descending so the largest (rarest multiples) are placed first
    indexed_divs = sorted(enumerate(divisors), key=lambda x: x[1], reverse=True)
    orig_indices = [d[0] for d in indexed_divs]
    sorted_divs = [d[1] for d in indexed_divs]

    # Precompute for each sorted divisor position: the list of multiples in [1, LCM]
    # For a window [L, L+span], a multiple m is in range if L <= m <= L+span
    # i.e., m - span <= L <= m
    # So for a given L, the valid multiples for divisor d are those m where m-L is in [0, span]
    
    def dfs(idx, chosen_set, current_max, L, span):
        nonlocal best_span, best_reps

        if time.time() > deadline:
            return

        if idx == n:
            span_actual = current_max - L
            if span_actual < best_span:
                best_span = span_actual
                temp_reps = [0] * n
                for i, (oi, val) in enumerate(zip(orig_indices, sorted_divs)):
                    # We need to recover the actual values
                    # This approach is getting complicated; let's use a different strategy
                    pass
            return

        d = sorted_divs[idx]

        # For this divisor, find all multiples in [L, L + best_span - 1]
        # (we need strictly less than best_span to improve)
        k_start = (L + d - 1) // d
        k_end = (L + best_span - 1) // d
        
        for k in range(k_start, k_end + 1):
            val = k * d
            if val in chosen_set:
                continue
            
            chosen_set.add(val)
            new_max = current_max if current_max > val else val
            dfs(idx + 1, chosen_set, new_max, L, best_span)
            chosen_set.discard(val)

    # Alternative approach: iterate over possible spans, and for each span,
    # check if there's a valid window. Start from small spans.
    
    # For a window of span S (max - min = S), we need 10 distinct values
    # in some interval [L, L+S] where each value is a multiple of its divisor.
    
    # Key insight: the minimum value in the window must be a multiple of its divisor.
    # So we can iterate over which divisor's multiple is the minimum, and what that multiple is.
    
    # Let's try: for each possible span S starting from 9 (minimum for 10 distinct ints),
    # for each possible L (the minimum), check if we can assign all divisors.
    
    # But L can be any integer, not just a multiple. However, the minimum of the 10 values
    # must be one of the values, which is a multiple of its divisor.
    
    # So: iterate over S from 9 upward. For each S, iterate over all possible "anchor" points
    # which are multiples of any divisor in the range. For each anchor A (which will be the min),
    # try to assign the other 9 divisors to distinct multiples in [A, A+S].
    
    # This is still complex. Let's use a different strategy:
    # For each L in [1, LCM], and for each possible span S, do a quick feasibility check.
    
    # Actually, let's just do the DFS but with better pruning and iterative deepening on span.
    
    best_reps_found = False
    
    # Try increasing spans
    for target_span in range(9, 500):  # try spans from 9 to 500
        if time.time() > deadline:
            break
        
        found = False
        
        # For this target span, iterate over L
        # L must be such that there exist 10 distinct multiples in [L, L+target_span]
        # The min of the 10 values is some multiple m_i of divisor i, and m_i >= L
        # The max is some multiple m_j of divisor j, and m_j <= L + target_span
        
        # To be efficient, only consider L values that are "interesting"
        # L can be any integer, but the set of multiples in [L, L+S] only changes
        # when L crosses a multiple boundary.
        
        # For simplicity, iterate L from 1 to LCM, but skip if time is up
        for L in range(1, LCM + 1):
            if time.time() > deadline:
                found = False
                break
            
            # Quick check: for each divisor, is there at least one multiple in [L, L+target_span]?
            feasible = True
            for d in sorted_divs:
                k_start = (L + d - 1) // d
                k_end = (L + target_span) // d
                if k_start > k_end:
                    feasible = False
                    break
            
            if not feasible:
                continue
            
            # Try to find an assignment via backtracking
            chosen_vals = []
            chosen_set = set()
            
            def dfs_idx(idx, cur_max):
                nonlocal found, best_reps, best_span
                
                if time.time() > deadline:
                    return True  # signal to stop
                if found:
                    return True
                
                if idx == n:
                    span_actual = cur_max - L
                    if span_actual <= target_span:
                        found = True
                        temp_reps = [0] * n
                        for i, (oi, val) in enumerate(zip(orig_indices, chosen_vals)):
                            temp_reps[oi] = val
                        best_reps = temp_reps
                        best_span = span_actual
                    return True
                
                d = sorted_divs[idx]
                k_start = (L + d - 1) // d
                k_end = (L + target_span) // d
                
                for k in range(k_start, k_end + 1):
                    val = k * d
                    if val in chosen_set:
                        continue
                    
                    chosen_vals.append(val)
                    chosen_set.add(val)
                    new_max = cur_max if cur_max > val else val
                    
                    if dfs_idx(idx + 1, new_max):
                        return True
                    
                    chosen_vals.pop()
                    chosen_set.discard(val)
                
                return False
            
            dfs_idx(0, 0)
            
            if found:
                break
        
        if found:
            break
    
    if not found and best_span == float('inf'):
        # Fallback
        return divisors[:]
    
    return [int(x) for x in best_reps]
