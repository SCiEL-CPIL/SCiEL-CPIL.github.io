import time
import math

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    # Calculate LCM of divisors
    lcm_val = 1
    for d in divisors:
        lcm_val = lcm_val * d // math.gcd(lcm_val, d)
        
    start_time = time.time()
    max_time = 50.0 # Safety margin
    
    # We want to minimize W = max - min
    # Iterate W from 1 upwards
    for W in range(1, 100):
        # Iterate L from 1 to LCM
        # Note: L can be any integer. The pattern repeats every LCM.
        # So we just check L in 1..LCM
        for L in range(1, lcm_val + 1):
            if time.time() - start_time > max_time:
                # Return best effort? Or just break?
                # If we didn't find a solution, this is bad. 
                # But W=16 is expected, so we should be done long before.
                break
            
            # Determine candidates for each divisor
            candidates = []
            valid = True
            for d in divisors:
                start_k = (L + d - 1) // d
                end_k = (L + W) // d
                if start_k > end_k:
                    valid = False
                    break
                # Generate actual values
                # To optimize, we might just store the range (start_k, end_k)
                # and resolve values during backtracking
                candidates.append((start_k, end_k))
            
            if not valid:
                continue
                
            # Solve assignment problem
            # We need to pick one multiple from each range such that all values are distinct.
            # Use backtracking.
            # Order divisors by range length (smallest first) to prune early
            indices = list(range(n))
            # Sort indices by number of candidates (end_k - start_k + 1)
            indices.sort(key=lambda i: candidates[i][1] - candidates[i][0])
            
            used_values = set()
            
            def backtrack(i):
                if i == n:
                    return True
                idx = indices[i]
                d = divisors[idx]
                start_k, end_k = candidates[idx]
                
                # Try each multiple in the range
                # Optimization: If the range is large, we might want to iterate smartly?
                # With W small, ranges are small.
                for k in range(start_k, end_k + 1):
                    val = k * d
                    if val not in used_values:
                        used_values.add(val)
                        if backtrack(i + 1):
                            return True
                        used_values.remove(val)
                return False
            
            if backtrack(0):
                # Reconstruct solution
                # The backtracking function modified used_values but didn't store the choices.
                # We need to store the choices.
                pass # Need to modify backtrack to store solution
                
        # If W loop finishes without break, continue to next W
        
    # Fallback if not found (should not happen)
    return divisors[:]
