"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import reduce

P = {'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39], 'maximum': 1000000}

def construct():
    divisors = P['divisors']
    max_val = P['maximum']
    deadline = time.time() + 55  # safety margin before 90s limit
    
    best_result = None
    best_range = float('inf')
    n = len(divisors)
    
    # Precompute sorted divisors for faster access
    # Order divisors by size (smaller divisors have more candidates in a window)
    div_idx_sorted = sorted(range(n), key=lambda i: divisors[i])
    
    # Try increasing window sizes
    for W in range(1, 150):  # Try windows up to size 149
        if time.time() > deadline:
            break
        
        # For each W, search L values.
        # Key optimization: for a given W, L only matters modulo each divisor.
        # The combined period is LCM of all divisors, but we can search a smaller range.
        # For small W, the valid L values are dense, so a moderate range should suffice.
        
        # Search L from 0 to 30000 (covers multiple periods of smaller divisors)
        l_search_max = 30000
        
        for L in range(0, l_search_max + 1):
            if time.time() > deadline:
                break
            
            # Get candidates for each divisor
            candidates = [None] * n
            valid = True
            for i, d in enumerate(divisors):
                # Find multiples of d in [L, L+W]
                k_min = (L + d - 1) // d  # ceil(L/d)
                k_max = (L + W) // d
                if k_max < k_min:
                    valid = False
                    break
                # Generate candidates as a list
                candidates[i] = list(range(k_min * d, k_max * d + 1, d))
            
            if not valid:
                continue
            
            # Quick rejection: if any divisor has only 1 candidate and that candidate
            # is shared by another divisor with only 1 candidate, it's infeasible.
            # More generally, use Hall's theorem check (simplified).
            
            # Backtracking with MRV (minimum remaining values)
            order = sorted(range(n), key=lambda i: len(candidates[i]))
            
            used = set()
            assignment = [None] * n
            
            # Recursive backtracking
            def backtrack(idx):
                if idx == n:
                    return True
                d_idx = order[idx]
                cands = candidates[d_idx]
                for m in cands:
                    if m not in used:
                        used.add(m)
                        assignment[d_idx] = m
                        if backtrack(idx + 1):
                            return True
                        used.remove(m)
                        assignment[d_idx] = None
                return False
            
            if backtrack(0):
                actual_vals = [assignment[i] for i in range(n)]
                actual_range = max(actual_vals) - min(actual_vals)
                if actual_range < best_range:
                    best_range = actual_range
                    best_result = [int(assignment[i]) for i in range(n)]
                    if best_range == 1:
                        return best_result
        
        if time.time() > deadline:
            break
    
    if best_result is None:
        # Fallback: return the divisors themselves
        return [int(d) for d in divisors]
    
    return best_result

# EVOLVE-BLOCK-END
