"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
import functools

P = {'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39], 'maximum': 1000000}

def construct():
    divisors = P['divisors']
    n = len(divisors)
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    period = functools.reduce(lcm, divisors, 1)
    
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit
    
    # For each window size R, we check if there exists a starting position L
    # such that for every divisor d, there is at least one multiple of d in [L, L+R],
    # and we can assign distinct multiples to each divisor.
    
    # The key insight: the pattern repeats every LCM = 30030.
    # So we only need to check L in [0, 30029].
    
    # For efficiency, we can precompute for each divisor d and each offset o in [0, period-1],
    # whether there's a multiple of d in [o, o+R]. But this depends on R.
    
    # Alternative: For each R, iterate L and use backtracking.
    # But we can optimize by noting that for a given L and R, the candidate set for divisor d
    # is {k*d : k*d in [L, L+R]}.
    
    # Let's iterate R from the theoretical minimum (n-1=9) upwards.
    # For each R, iterate L from 0 to period-1.
    # Use backtracking to check if a valid assignment exists.
    
    best_result = None
    best_range = float('inf')
    
    for R in range(9, 200):  # Try window sizes from 9 to 199
        if time.time() > deadline:
            break
        
        # For this R, check all L in [0, period-1]
        found = False
        
        # Optimization: We can skip L values that are equivalent modulo some divisors
        # But for now, let's just iterate all L.
        
        for L in range(period):
            if time.time() > deadline:
                break
            
            # Generate candidates for each divisor
            candidates = []
            valid = True
            
            for d in divisors:
                # Multiples of d in [L, L+R]
                k_min = (L + d - 1) // d
                k_max = (L + R) // d
                
                if k_max < k_min:
                    valid = False
                    break
                
                cands = [k * d for k in range(k_min, k_max + 1)]
                candidates.append(cands)
            
            if not valid:
                continue
            
            # Backtracking with MRV (Minimum Remaining Values)
            indices = list(range(n))
            indices.sort(key=lambda i: len(candidates[i]))
            
            used = set()
            assignment = [None] * n
            
            def backtrack(idx):
                if idx == n:
                    return True
                d_idx = indices[idx]
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
                # Found a valid assignment
                actual_vals = [assignment[i] for i in range(n)]
                actual_range = max(actual_vals) - min(actual_vals)
                
                if actual_range < best_range:
                    best_range = actual_range
                    best_result = [int(v) for v in actual_vals]
                
                found = True
                break
        
        if found:
            # Since we're iterating R from small to large, and we found a solution,
            # this is the optimal range (or very close to it).
            # Actually, we should continue to check if a smaller range exists?
            # No, because if a solution with range < R exists, it would have been found
            # in a previous iteration with R' < R.
            # So we can return.
            return best_result
    
    if best_result is None:
        # Fallback: return the divisors themselves (not optimal but valid)
        return [int(d) for d in divisors]
    
    return best_result

# EVOLVE-BLOCK-END
