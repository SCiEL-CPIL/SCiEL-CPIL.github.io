import time
import bisect

P = {'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39], 'maximum': 1000000}

def construct():
    divisors = P['divisors']
    max_val = P['maximum']
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Precompute multiples for each divisor
    multiples = {}
    for d in divisors:
        multiples[d] = list(range(d, max_val + 1, d))
    
    # We search for the minimum window width W
    # We iterate W from 0 upwards
    best_solution = None
    best_W = float('inf')
    
    # Upper bound for W search
    # A trivial solution is to take the first 10 distinct multiples for each?
    # Or just take multiples far apart.
    # But we expect W to be small.
    # Let's try up to 500.
    
    for W in range(0, 500):
        if time.time() > deadline:
            break
            
        # Generate candidate L values
        # L is a candidate if it is a critical point for coverage
        # Critical points are k*d and k*d - W
        candidates = set()
        for d in divisors:
            # Multiples in range [1, max_val]
            # We need L such that [L, L+W] is within [1, max_val]
            # So L in [1, max_val - W]
            # Critical points for d:
            # k*d and k*d - W
            # k*d must be in [1, max_val]
            # k*d - W must be in [1, max_val - W] -> k*d in [1+W, max_val]
            
            # We can just generate all multiples and their -W shifts
            for m in multiples[d]:
                candidates.add(m)
                candidates.add(m - W)
        
        # Filter candidates to valid range [1, max_val - W]
        candidates = [L for L in candidates if 1 <= L <= max_val - W]
        
        # To avoid checking too many, we can sort and maybe sample?
        # But for small W, the number of candidates is small?
        # No, it's sum of 10^6/d. ~800k.
        # Checking 800k candidates for each W is too slow if W goes up to 500.
        
        # Optimization: Only check candidates where ALL divisors have at least one multiple.
        # We can do a quick check.
        
        # Better: Iterate through candidates, but break early if found.
        # Also, if W is small, valid L's are rare.
        
        # Let's try to be smarter.
        # If we find a solution, we return.
        
        # We will iterate through candidates.
        # To speed up, we can sort candidates and check them.
        # But which order?
        # Random order might find a solution faster if they are scattered.
        # Sorted order might find them if they are clustered.
        # Let's just iterate.
        
        # To prevent timeout, we limit the number of candidates checked per W?
        # Or rely on the fact that if a solution exists, it's likely to be found quickly?
        
        # Actually, if W is the optimal width, there might be many solutions.
        # If W < optimal, there are none.
        
        # Let's just try.
        
        # Sort candidates to be deterministic? Or random?
        # Let's just use the set.
        
        # Optimization: If W is large, the number of candidates is huge.
        # But we expect W to be small.
        
        # Let's add a check: if time is running out, break.
        
        for L in candidates:
            if time.time() > deadline:
                break
                
            # Check if valid assignment exists in [L, L+W]
            sol = _solve_window(L, W, divisors, multiples)
            if sol is not None:
                return sol
        
        # If no solution found for this W, continue to next W
    # If we get here, return best found or a fallback
    # Fallback: smallest multiples
    return P['divisors'][:]

def _solve_window(L, W, divisors, multiples):
    # Get available multiples for each divisor in [L, L+W]
    available = []
    for d in divisors:
        m_list = multiples[d]
        # Find indices in m_list for range [L, L+W]
        left_idx = bisect.bisect_left(m_list, L)
        right_idx = bisect.bisect_right(m_list, L + W)
        if left_idx >= right_idx:
            return None # No multiple for this divisor
        avail = m_list[left_idx:right_idx]
        available.append(avail)
    
    # Backtracking with MRV
    n = len(divisors)
    assignment = [None] * n
    
    # Order variables by domain size
    # We need to track which divisor index corresponds to which available list
    # available[i] is for divisors[i]
    
    # Create list of (domain_size, divisor_index)
    order = sorted(range(n), key=lambda i: len(available[i]))
    
    def backtrack(idx):
        if idx == n:
            return True
        
        # Get the next variable to assign
        var_idx = order[idx]
        
        # Try each value in the domain
        for val in available[var_idx]:
            # Check if val is already used
            if val in assignment:
                continue
            
            assignment[var_idx] = val
            
            if backtrack(idx + 1):
                return True
            
            assignment[var_idx] = None
            
        return False
    
    if backtrack(0):
        return assignment
    return None
