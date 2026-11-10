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
    # We will iterate W from 9 upwards
    for W in range(9, 300):
        if time.time() - start_time > timeout:
            # Fallback: return a valid solution
            return divisors[:]
        
        # For this window width W, we need to find a starting position L
        # such that we can assign distinct multiples.
        # Key insight: L only matters modulo the LCM, and we can skip many L values.
        # For a given L, the candidates for divisor d are multiples in [L, L+W].
        # The first multiple >= L is ceil(L/d)*d.
        # As L increases by 1, the candidate set changes only when L crosses a multiple boundary.
        # So we only need to check L values that are "interesting": L = k*d for some d, or L = k*d - 1.
        
        # Build set of interesting L values: for each divisor d, values k*d and k*d-1 
        # that fall in [1, LCM]
        interesting_L = set()
        for d in divisors:
            for k in range(1, LCM // d + 1):
                interesting_L.add(k * d)
                if k * d - 1 >= 1:
                    interesting_L.add(k * d - 1)
        # Also add 1 and LCM
        interesting_L.add(1)
        interesting_L.add(LCM)
        
        # Sort for deterministic processing
        interesting_L = sorted(interesting_L)
        
        for L in interesting_L:
            if time.time() - start_time > timeout:
                return divisors[:]
            
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
