import math

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    # Heuristic: Sort divisors by their value? 
    # Actually, MRV (Minimum Remaining Values) is better during search.
    # But for the outer loop, we just need to find ANY valid M.
    
    def solve_for_range(R):
        # We search for M in a reasonable range.
        # If a solution exists, it likely exists for small M.
        # The pattern of multiples repeats with LCM, but we don't need to go that far.
        # Let's try M up to 5000.
        max_M = 5000
        
        for M in range(1, max_M + 1):
            # Get candidates for each divisor
            candidates = []
            valid = True
            for d in divisors:
                # Multiples of d in [M, M+R]
                k_start = (M + d - 1) // d
                k_end = (M + R) // d
                mults = []
                for k in range(k_start, k_end + 1):
                    val = k * d
                    if M <= val <= M + R:
                        mults.append(val)
                if not mults:
                    valid = False
                    break
                candidates.append(mults)
            
            if not valid:
                continue
            
            # Backtracking with MRV heuristic
            # We need to assign a value to each divisor index i
            # from candidates[i] such that all values are distinct.
            
            # To use MRV, we should pick the variable with the smallest domain first.
            # Since the domains change as we assign values, we do this dynamically.
            
            assignment = [None] * n
            used = set()
            
            # Indices of divisors
            indices = list(range(n))
            
            def backtrack(remaining_indices):
                if not remaining_indices:
                    return True
                
                # Pick index with minimum remaining values (MRV)
                # We need to filter out used values for each candidate
                best_idx = -1
                best_size = float('inf')
                best_opts = []
                
                for i in remaining_indices:
                    opts = [v for v in candidates[i] if v not in used]
                    if not opts:
                        return False
                    if len(opts) < best_size:
                        best_size = len(opts)
                        best_idx = i
                        best_opts = opts
                        if best_size == 1:
                            break
                
                # Remove best_idx from remaining
                new_remaining = [i for i in remaining_indices if i != best_idx]
                
                # Try each option
                for v in best_opts:
                    used.add(v)
                    assignment[best_idx] = v
                    if backtrack(new_remaining):
                        return True
                    used.remove(v)
                    assignment[best_idx] = None
                
                return False
            
            if backtrack(indices):
                # Verify
                if len(set(assignment)) == n and all(1 <= x <= 1000000 for x in assignment):
                    return assignment[:]
        
        return None

    # Binary search for minimum R
    # Lower bound: 9 (since 10 distinct integers)
    # Upper bound: Let's start with 1000
    low = 9
    high = 1000
    best_sol = None
    
    # We want the smallest R.
    # We can just iterate R from 9 upwards until we find a solution.
    # Binary search is valid if the property "solvable with range R" is monotonic.
    # If solvable with R, is it solvable with R+1?
    # Yes, if we have a set of 10 distinct integers in [M, M+R], they are also in [M, M+R+1].
    # So the property is monotonic.
    # However, we want the MINIMUM R.
    # So we can binary search for the smallest R such that solve_for_range(R) is not None.
    
    while low < high:
        mid = (low + high) // 2
        sol = solve_for_range(mid)
        if sol is not None:
            best_sol = sol
            high = mid
        else:
            low = mid + 1
    
    if best_sol is None:
        # Fallback: try larger ranges or just return divisors
        # Try up to 5000
        for R in range(1001, 5001):
            sol = solve_for_range(R)
            if sol is not None:
                best_sol = sol
                break
        if best_sol is None:
            return divisors[:]
            
    return best_sol
