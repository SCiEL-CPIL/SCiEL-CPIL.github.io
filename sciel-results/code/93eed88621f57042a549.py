"""
Strategy: Iterate over the minimum value (v_min) and use a greedy 'minimize max' assignment for the remaining divisors.
This avoids the heuristic window sliding and directly optimizes the span for each potential start point.
"""
import time

# Fixed inputs
DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
N = len(DIVISORS)
MAX_VAL = 1000000

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # 50s safety margin
    
    # Seed solution: the divisors themselves
    # Max: 39, Min: 6, Span: 33
    best_span = 33
    best_vals = DIVISORS[:]
    
    # We iterate over possible v_min values.
    # Since we want small spans, and the problem is periodic, 
    # checking v_min up to 100,000 should be sufficient and fast enough.
    # We can optimize by only checking v_min that are multiples of at least one divisor.
    
    # Precompute multiples? No, just iterate.
    
    # Optimization: If we find a span of 9 (10 distinct integers), we can stop.
    MIN_POSSIBLE_SPAN = 9
    
    # To speed up, we can jump v_min to the next multiple of any divisor?
    # Or just iterate all integers? 
    # Iterating all integers 1..100000 is 100k steps.
    # Inner check: 
    #   Find anchors (divisors dividing v_min).
    #   For each anchor, generate candidates for others.
    #   Greedy assign.
    # This is fast.
    
    for v_min in range(1, 100001):
        if time.time() > deadline:
            break
        
        # Check if v_min is a multiple of any divisor
        anchors = []
        for i, d in enumerate(DIVISORS):
            if v_min % d == 0:
                anchors.append(i)
        
        if not anchors:
            continue
            
        # For each possible anchor (divisor assigned to v_min)
        for k in anchors:
            # We assign v_min to divisor k.
            # We need to assign distinct multiples to the other 9 divisors.
            # To minimize the max, we gather all potential candidates for the other divisors
            # within the range [v_min + 1, v_min + best_span].
            
            # If the range is empty (best_span == 0), skip.
            if best_span == 0:
                break
                
            upper_bound = v_min + best_span
            
            # List of (value, divisor_index) for all candidates
            candidates = []
            valid_anchor = True
            
            for j in range(N):
                if j == k:
                    continue
                d = DIVISORS[j]
                # Find multiples of d in (v_min, upper_bound]
                # First multiple > v_min
                m = ((v_min // d) + 1) * d
                while m <= upper_bound:
                    candidates.append((m, j))
                    m += d
                # If no candidates found, this anchor cannot yield a solution better than current best
                # (because we couldn't even find a value within the current best span)
                # Note: It's possible that a value > upper_bound is needed, but that would result in a span > best_span, 
                # so we don't care for the purpose of improving the best.
                if not any(c[1] == j for c in candidates):
                    valid_anchor = False
                    break
            
            if not valid_anchor:
                continue
            
            # Sort candidates by value
            candidates.sort()
            
            # Greedy assignment: 
            # Iterate through sorted candidates.
            # If the divisor is not yet assigned, assign it.
            assigned = [False] * N
            assigned[k] = True
            vals = [0] * N
            vals[k] = v_min
            
            assigned_count = 1
            max_val = v_min
            
            for val, div_idx in candidates:
                if assigned[div_idx]:
                    continue
                # Assign val to div_idx
                assigned[div_idx] = True
                vals[div_idx] = val
                assigned_count += 1
                if val > max_val:
                    max_val = val
                
                if assigned_count == N:
                    break
            
            if assigned_count == N:
                span = max_val - v_min
                if span < best_span:
                    best_span = span
                    best_vals = vals[:]
                    if best_span == MIN_POSSIBLE_SPAN:
                        return [int(v) for v in best_vals]
    
    # Fallback to seed if no better found (or if we timed out)
    # Ensure we return the best found
    if best_span < 33:
        return [int(v) for v in best_vals]
    else:
        return DIVISORS[:]

# EVOLVE-BLOCK-END
