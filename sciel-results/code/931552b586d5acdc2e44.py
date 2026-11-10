import time
from math import gcd
from functools import reduce

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    LCM = reduce(lcm, divisors)  # 30030
    
    start_time = time.time()
    timeout = 85  # Safety margin for 90s limit
    
    # We search for minimal W (max - min)
    # Minimum W is 9 (10 distinct integers)
    # Key insight: the minimum value in any optimal solution must be a multiple
    # of at least one divisor. So we only need to check L values that are multiples
    # of some divisor, plus L-1 for boundary cases.
    
    # Precompute all multiples of each divisor up to LCM
    multiples_by_divisor = []
    for d in divisors:
        multiples = list(range(d, LCM + 1, d))
        multiples_by_divisor.append(multiples)
    
    # Collect all "interesting" L values: multiples of divisors and their predecessors
    interesting_L = set()
    for multiples in multiples_by_divisor:
        for m in multiples:
            interesting_L.add(m)
            if m > 1:
                interesting_L.add(m - 1)
    interesting_L = sorted(interesting_L)
    
    # For each W from 9 upward, check if there's a valid window
    best_result = None
    for W in range(9, 200):
        if time.time() - start_time > timeout:
            break
        
        for L in interesting_L:
            if time.time() - start_time > timeout:
                break
            
            # For this window [L, L+W], determine candidates for each divisor
            forced = [None] * n
            flexible = []
            valid_window = True
            
            for i, d in enumerate(divisors):
                # Find first multiple >= L
                k = (L + d - 1) // d
                c1 = k * d
                
                if c1 > L + W:
                    valid_window = False
                    break
                
                # Collect all multiples of d in [L, L+W]
                candidates = []
                x = c1
                while x <= L + W:
                    candidates.append(x)
                    x += d
                
                if len(candidates) == 1:
                    forced[i] = candidates[0]
                else:
                    flexible.append((i, candidates))
            
            if not valid_window:
                continue
            
            # Check for conflicts in forced assignments
            used_forced = set()
            has_conflict = False
            for i in range(n):
                if forced[i] is not None:
                    if forced[i] in used_forced:
                        has_conflict = True
                        break
                    used_forced.add(forced[i])
            
            if has_conflict:
                continue
            
            # Now we need to assign values to flexible divisors
            # Sort flexible by number of candidates (most constrained first)
            flexible.sort(key=lambda x: len(x[1]))
            
            # Backtracking search
            result = [0] * n
            for i in range(n):
                if forced[i] is not None:
                    result[i] = forced[i]
            
            used = set(used_forced)
            
            def backtrack(f_idx):
                if f_idx == len(flexible):
                    return True
                idx, candidates = flexible[f_idx]
                for val in candidates:
                    if val not in used:
                        used.add(val)
                        result[idx] = val
                        if backtrack(f_idx + 1):
                            return True
                        used.remove(val)
                return False
            
            if backtrack(0):
                # Verify
                if len(set(result)) == n:
                    for i in range(n):
                        if result[i] % divisors[i] != 0 or not (0 < result[i] <= 1000000):
                            break
                    else:
                        return result
    
    # Fallback if no solution found in time or range
    return divisors[:]
