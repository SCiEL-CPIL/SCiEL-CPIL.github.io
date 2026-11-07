import math
from functools import lru_cache

# EVOLVE-BLOCK-START
DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000

def construct():
    # Precompute LCM of divisors
    K = 1
    for d in DIVISORS:
        K = math.lcm(K, d)
    
    # Function to check if a solution exists for a given range R
    def can_solve(R):
        # We iterate L from 1 to K. 
        # The pattern of multiples repeats every K.
        # If a solution exists for some L, it exists for L % K (adjusted for positivity).
        # Since x_i >= L >= 1, and x_i <= L+R, and R is small, 
        # checking L in [1, K] covers all relative alignments.
        
        # Optimization: We can skip many L if domains are empty.
        # But constructing domains is fast.
        
        for L in range(1, K + 1):
            # Build domains
            domains = []
            valid = True
            for d in DIVISORS:
                # Smallest multiple of d >= L
                # m = ceil(L/d)
                m = (L + d - 1) // d
                start_val = m * d
                # Largest multiple of d <= L+R
                # m2 = floor((L+R)/d)
                m2 = (L + R) // d
                end_val = m2 * d
                
                if start_val > end_val:
                    valid = False
                    break
                
                # Generate list of values
                # Range is small, so list comprehension is fine
                # Values are start_val, start_val+d, ..., end_val
                # But we only need distinctness across variables.
                # The number of values is at most (R/d) + 1, which is small.
                vals = []
                v = start_val
                while v <= end_val:
                    vals.append(v)
                    v += d
                domains.append(vals)
            
            if not valid:
                continue
                
            # Backtracking search
            # Sort variables by domain size (MRV)
            # We need to map back to original indices for the result
            indices = list(range(len(DIVISORS)))
            indices.sort(key=lambda i: len(domains[i]))
            
            used_vals = set()
            
            def backtrack(idx):
                if idx == len(indices):
                    return True
                
                var_idx = indices[idx]
                for val in domains[var_idx]:
                    if val not in used_vals:
                        used_vals.add(val)
                        if backtrack(idx + 1):
                            return True
                        used_vals.remove(val)
                return False
            
            if backtrack(0):
                # Found a solution for this L and R
                # We need to retrieve the actual values.
                # The backtrack function above just returns True/False.
                # We need to modify it to store the assignment or return it.
                # For efficiency in 'can_solve', we just need True/False.
                # For the final answer, we will re-run or store it.
                # To avoid re-running, we can have can_solve return the assignment.
                # But binary search calls it many times. 
                # Let's make a separate function to find the solution once R is known.
                # Or, just return True here.
                return True
        
        return False

    # Binary search for minimal R
    # Lower bound: 1 (distinct integers)
    # Upper bound: Let's estimate. 
    # Worst case, we might need a larger range. 
    # Heuristic: The gap between consecutive multiples of different divisors can be large.
    # But with 10 variables, the range is likely small.
    # Let's set high = 1000. If not found, expand.
    
    low = 1
    high = 1000
    
    # Ensure high is feasible
    while not can_solve(high):
        high *= 2
        if high > 100000: # Safety break
            break
            
    while low < high:
        mid = (low + high) // 2
        if can_solve(mid):
            high = mid
        else:
            low = mid + 1
            
    # Now low is the minimal R
    # Find the actual assignment for R = low
    R_best = low
    
    # Re-run the search to get the values
    for L in range(1, K + 1):
        domains = []
        valid = True
        for d in DIVISORS:
            m = (L + d - 1) // d
            start_val = m * d
            m2 = (L + R_best) // d
            end_val = m2 * d
            
            if start_val > end_val:
                valid = False
                break
            
            vals = []
            v = start_val
            while v <= end_val:
                vals.append(v)
                v += d
            domains.append(vals)
        
        if not valid:
            continue
            
        indices = list(range(len(DIVISORS)))
        indices.sort(key=lambda i: len(domains[i]))
        
        used_vals = set()
        assignment = [None] * len(DIVISORS)
        
        def backtrack(idx):
            if idx == len(indices):
                return True
            
            var_idx = indices[idx]
            for val in domains[var_idx]:
                if val not in used_vals:
                    used_vals.add(val)
                    assignment[var_idx] = val
                    if backtrack(idx + 1):
                        return True
                    used_vals.remove(val)
                    assignment[var_idx] = None
            return False
            
        if backtrack(0):
            # Verify constraints
            # Distinct? Yes, by construction.
            # Divisible? Yes, by domain construction.
            # Range? max - min <= R_best.
            # Positive? L >= 1, so x_i >= 1.
            # <= 1000000? L <= K ~ 30000, R small, so yes.
            
            # Convert to list of ints
            result = [int(x) for x in assignment]
            return result
            
    # Fallback if something went wrong (should not happen)
    # Return divisors themselves (which might not be distinct or minimal range, but valid types)
    return DIVISORS[:]

# EVOLVE-BLOCK-END
