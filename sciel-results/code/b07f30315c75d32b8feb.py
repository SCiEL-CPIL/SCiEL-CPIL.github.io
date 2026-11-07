import math
import time

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    max_val = 1000000
    
    # Calculate LCM of divisors
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    period = 1
    for d in divisors:
        period = lcm(period, d)
        
    # period is 32730
    
    # We want to minimize R = max - min.
    # The values must be distinct.
    # We search for the smallest R such that there exists L 
    # where we can pick distinct multiples v_i in [L, L+R-1] 
    # with v_i % d_i == 0.
    
    # Since the pattern repeats every 'period', we only need to check 
    # L in [0, period-1].
    
    # We iterate R from 1 upwards.
    # For each R, we iterate L from 0 to period-1.
    # To speed up, we can break early if we find a solution.
    
    # Optimization: 
    # For a given R, if for some divisor d_i, the number of multiples 
    # in [L, L+R-1] is 0, it's impossible.
    # Also, if the total number of candidates is less than n, impossible.
    
    # Let's use a backtracking search with MRV heuristic.
    
    def can_assign(L, R):
        # candidates[i] is list of multiples of divisors[i] in [L, L+R-1]
        candidates = []
        for i in range(n):
            d = divisors[i]
            # first multiple >= L
            k_start = (L + d - 1) // d
            # last multiple <= L+R-1
            k_end = (L + R - 1) // d
            
            if k_start > k_end:
                return None
            
            # Generate candidates
            # If R is small, this list is short
            # max size is roughly R/d + 1
            cands = []
            for k in range(k_start, k_end + 1):
                val = k * d
                cands.append(val)
            candidates.append(cands)
            
        # Check if any candidate list is empty
        for c in candidates:
            if not c:
                return None
                
        # Backtracking with MRV
        # Order indices by length of candidates
        order = sorted(range(n), key=lambda i: len(candidates[i]))
        assigned = [None] * n
        used = set()
        
        def backtrack(idx):
            if idx == n:
                return True
            i = order[idx]
            for val in candidates[i]:
                if val not in used:
                    used.add(val)
                    assigned[i] = val
                    if backtrack(idx + 1):
                        return True
                    used.remove(val)
                    assigned[i] = None
            return False
        
        if backtrack(0):
            return assigned
        return None

    start_time = time.time()
    deadline = start_time + 85  # 85s out of 90s to be safe
    
    # We expect R to be small. Let's try up to 200.
    # If we don't find it, we might need to increase, but 200 should be enough 
    # for this set of divisors.
    
    for R in range(1, 300):
        if time.time() > deadline:
            break
            
        # Optimization: If R is very small, many L will fail immediately.
        # We can check L from 0 to period-1.
        
        for L in range(period):
            if time.time() > deadline:
                break
                
            # Quick check: count total candidates
            # If total candidates < n, skip
            total_cands = 0
            valid = True
            for i in range(n):
                d = divisors[i]
                k_start = (L + d - 1) // d
                k_end = (L + R - 1) // d
                if k_start > k_end:
                    valid = False
                    break
                total_cands += (k_end - k_start + 1)
                if total_cands >= n and (k_end - k_start + 1) == 0:
                    valid = False
                    break
            if not valid or total_cands < n:
                continue
                
            res = can_assign(L, R)
            if res is not None:
                # Check if values are within max_val
                # Since L < period and R < 300, max val is < 32730 + 300 < 1000000
                # So it's always valid
                return [int(res[i]) for i in range(n)]
                
    # Fallback: return divisors themselves (distinct, valid, but large range)
    return [int(d) for d in divisors]
