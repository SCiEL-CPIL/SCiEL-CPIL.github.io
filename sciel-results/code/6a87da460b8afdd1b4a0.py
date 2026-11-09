import time
from math import gcd
from functools import reduce

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    max_val = 1000000
    n = len(divisors)
    
    # Fallback: the divisors themselves (span 33)
    fallback = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # LCM of all divisors
    def lcm(a, b):
        return a * b // gcd(a, b)
    M = reduce(lcm, divisors)  # 270270
    
    # Iterate W from minimum possible (9) upward
    for W in range(9, 100):
        if time.time() > deadline:
            break
            
        # For each starting position L modulo M, check if a valid assignment exists
        # in the window [L, L+W]. Due to periodicity, we only need L in [0, M-1].
        # But we also need the final values to be in [1, max_val].
        
        found = False
        for L in range(M):
            if time.time() > deadline:
                break
                
            # For each divisor, find multiples in [L, L+W]
            candidates = []
            valid = True
            for i, d in enumerate(divisors):
                # Find k such that L <= k*d <= L+W
                k_min = (L + d - 1) // d  # ceil(L/d)
                k_max = (L + W) // d       # floor((L+W)/d)
                
                if k_min > k_max:
                    valid = False
                    break
                
                vals = [k * d for k in range(k_min, k_max + 1)]
                candidates.append(vals)
            
            if not valid:
                continue
            
            # Check if we can assign distinct values to each divisor
            # Sort by number of candidates (most constrained first)
            order = sorted(range(n), key=lambda i: len(candidates[i]))
            
            # Quick check: if any candidate list is empty, skip
            if any(len(candidates[i]) == 0 for i in range(n)):
                continue
            
            # Backtracking with MRV
            used = set()
            assignment = [None] * n
            
            def backtrack(idx):
                if time.time() > deadline:
                    return False
                if idx == n:
                    return True
                i = order[idx]
                for v in candidates[i]:
                    if v not in used:
                        used.add(v)
                        assignment[i] = v
                        if backtrack(idx + 1):
                            return True
                        used.remove(v)
                        assignment[i] = None
                return False
            
            if backtrack(0):
                # Found a valid set of values in [L, L+W]
                # Now shift to fit in [1, max_val]
                v_min = min(assignment)
                v_max = max(assignment)
                
                # Find t such that 1 <= v_i + t*M <= max_val for all i
                # t >= ceil((1 - v_min) / M) and t <= floor((max_val - v_max) / M)
                t_min = 0
                if v_min < 1:
                    t_min = (1 - v_min + M - 1) // M  # ceil((1-v_min)/M)
                
                t_max = (max_val - v_max) // M
                
                if t_min <= t_max:
                    t = t_min
                    final_assign = [assignment[i] + t * M for i in range(n)]
                    
                    # Verify
                    ok = True
                    for i, d in enumerate(divisors):
                        if final_assign[i] < 1 or final_assign[i] > max_val:
                            ok = False
                            break
                        if final_assign[i] % d != 0:
                            ok = False
                            break
                    if len(set(final_assign)) != n:
                        ok = False
                    
                    if ok:
                        return [int(v) for v in final_assign]
                
                # If no valid t, continue searching
                found = True  # We found a pattern, but couldn't shift it
        
        if found:
            # Found a pattern for this W but couldn't shift? Unlikely. Continue to next W.
            # Actually, if we found a pattern, we should be able to shift it since W < max_val.
            # If not, something is wrong. Let's continue.
            pass
    
    return fallback
