"""
Strategy: Iterate over the minimum value (v_min) and use a backtracking search to assign distinct multiples
within a window [v_min, v_min + best_span]. This ensures distinctness and optimizes the span.
"""
import time

# Fixed inputs
DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
N = len(DIVISORS)
MAX_VAL = 1000000

def solve_window(v_min, best_span, anchor_idx):
    """
    Try to assign distinct multiples to all divisors such that:
    - Divisor anchor_idx is assigned v_min.
    - All other values are in (v_min, v_min + best_span].
    - All values are distinct.
    Returns a list of values if found, else None.
    """
    vals = [None] * N
    vals[anchor_idx] = v_min
    used = {v_min}
    
    # Get list of other divisors
    others = [j for j in range(N) if j != anchor_idx]
    
    # To improve performance, we can sort 'others' by how constrained they are?
    # Or just iterate. The window is small.
    
    def backtrack(idx):
        if idx == len(others):
            return True
        
        j = others[idx]
        d = DIVISORS[j]
        # Start from v_min + 1
        # First multiple of d > v_min
        start = ((v_min // d) + 1) * d
        end = v_min + best_span
        
        # Try all multiples in range
        m = start
        while m <= end:
            if m not in used:
                used.add(m)
                vals[j] = m
                if backtrack(idx + 1):
                    return True
                used.remove(m)
                vals[j] = None
            m += d
        return False

    if backtrack(0):
        return vals
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # 55s safety margin
    
    # Seed solution: the divisors themselves
    # Max: 39, Min: 6, Span: 33
    best_span = 33
    best_vals = DIVISORS[:]
    
    # We iterate over possible v_min values.
    # We can limit the range. Since span is small, v_min doesn't need to be huge
    # to find a small span, but periodicity means we should check a range.
    # LCM of divisors is large, but local patterns repeat.
    # Checking up to 100,000 is safe.
    
    MIN_POSSIBLE_SPAN = 9
    
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
            # Try to find a valid assignment within current best_span
            # If found, it must be strictly better (span < best_span) or equal?
            # If equal, we don't need to update, but we might find a "better" one later?
            # Actually, if we find a solution with span < best_span, we update.
            
            # Optimization: If best_span is already minimal, break.
            if best_span == MIN_POSSIBLE_SPAN:
                return [int(v) for v in best_vals]
                
            # The window is (v_min, v_min + best_span] for others.
            # Wait, if we find a solution, the max value will be <= v_min + best_span.
            # The actual span is max(vals) - v_min.
            # We want to see if we can fit them in a span < best_span?
            # No, the backtracking checks if they fit in [v_min+1, v_min+best_span].
            # If they fit, the resulting span is <= best_span.
            # To improve best_span, we need the resulting span to be < best_span.
            
            # However, the backtracking as written allows values up to v_min + best_span.
            # If it finds a solution, the max value is some M <= v_min + best_span.
            # The span is M - v_min.
            # If M - v_min < best_span, we have an improvement.
            # If M - v_min == best_span, it's not an improvement.
            
            # To ensure we find improvements, we can just run the solver.
            # If it returns a solution, calculate the actual span.
            
            res = solve_window(v_min, best_span, k)
            if res:
                current_max = max(res)
                current_span = current_max - v_min
                if current_span < best_span:
                    best_span = current_span
                    best_vals = res[:]
                    if best_span == MIN_POSSIBLE_SPAN:
                        return [int(v) for v in best_vals]
    
    # Fallback to seed if no better found (or if we timed out)
    if best_span < 33:
        return [int(v) for v in best_vals]
    else:
        return DIVISORS[:]

# EVOLVE-BLOCK-END
