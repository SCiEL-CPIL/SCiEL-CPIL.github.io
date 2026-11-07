import time
import math
import numpy as np

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    start_time = time.time()
    deadline = start_time + 85.0
    
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    lcm_val = 1
    for d in divisors:
        lcm_val = lcm(lcm_val, d)
    
    # Known good solution with range 19
    known_good = [24, 40, 28, 30, 21, 22, 26, 33, 35, 39]
    best_range = max(known_good) - min(known_good)
    best_vals = known_good[:]
    
    # Precompute multiples for each divisor as sorted lists
    multiples_by_divisor = []
    for d in divisors:
        mults = list(range(0, lcm_val, d))
        multiples_by_divisor.append(mults)
    
    # Search for better solutions
    for W in range(9, best_range):
        if time.time() > deadline:
            break
        
        found = False
        for A in range(0, lcm_val):
            if time.time() > deadline:
                break
            
            candidates = []
            ok = True
            for d_idx, d in enumerate(divisors):
                mults = multiples_by_divisor[d_idx]
                
                # Find multiples in circular window [A, A+W] mod lcm_val
                if A + W < lcm_val:
                    # Non-wrapping
                    lo = A
                    hi = A + W
                    # Binary search
                    import bisect
                    left = bisect.bisect_left(mults, lo)
                    right = bisect.bisect_right(mults, hi)
                    if left >= right:
                        ok = False
                        break
                    candidates.append(mults[left:right])
                else:
                    # Wrapping
                    import bisect
                    # Part 1: [A, lcm_val)
                    left1 = bisect.bisect_left(mults, A)
                    right1 = len(mults)
                    # Part 2: [0, (A+W) % lcm_val]
                    wrap_end = (A + W) % lcm_val
                    left2 = 0
                    right2 = bisect.bisect_right(mults, wrap_end)
                    
                    if left1 >= right1 and left2 >= right2:
                        ok = False
                        break
                    if left1 >= right1:
                        candidates.append(mults[left2:right2])
                    elif left2 >= right2:
                        candidates.append(mults[left1:right1])
                    else:
                        candidates.append(mults[left1:right1] + mults[left2:right2])
            
            if not ok:
                continue
            
            # Prune: if any divisor has too many candidates, skip
            min_count = min(len(c) for c in candidates)
            if min_count > 3:
                continue
            
            # Backtracking with MRV heuristic
            order = sorted(range(n), key=lambda i: len(candidates[i]))
            
            used = set()
            assignment = [None] * n
            
            def bt(idx):
                if idx == n:
                    return True
                i = order[idx]
                for v in candidates[i]:
                    if v not in used:
                        used.add(v)
                        assignment[i] = v
                        if bt(idx + 1):
                            return True
                        used.discard(v)
                        assignment[i] = None
                return False
            
            if bt(0):
                vals = assignment[:]
                r = max(vals) - min(vals)
                if r < best_range:
                    best_range = r
                    best_vals = vals[:]
                    found = True
        
        if best_range <= W:
            break
    
    return [int(v) for v in best_vals]
