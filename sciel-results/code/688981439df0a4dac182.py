import time
from math import gcd

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    
    # LCM of all divisors
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    period = 1
    for d in divisors:
        period = lcm(period, d)
    # period = 30030
    
    best_span = float('inf')
    best_result = None
    
    for L in range(period):
        if time.time() > deadline:
            break
        
        # For each divisor, compute the first multiple >= L
        firsts = []
        for d in divisors:
            k = (L + d - 1) // d
            firsts.append(k * d)
        
        R_min = max(firsts)
        
        # Try increasing R from R_min to find feasible assignment
        for R in range(R_min, R_min + 200):
            if time.time() > deadline:
                break
            
            # Build options for each divisor in [L, R]
            options = []
            feasible_all = True
            for i, d in enumerate(divisors):
                k_start = (L + d - 1) // d
                k_end = R // d
                if k_end >= k_start:
                    opts = list(range(k_start * d, k_end * d + 1, d))
                else:
                    opts = []
                    feasible_all = False
                    break
                options.append((len(opts), i, opts))
            
            if not feasible_all:
                continue
            
            # Sort by number of options (MRV heuristic)
            options.sort(key=lambda x: x[0])
            
            # Backtracking with MRV
            used = set()
            assignment = [None] * n
            
            def backtrack(idx):
                if time.time() > deadline:
                    return False
                if idx == len(options):
                    return True
                _, i, opts = options[idx]
                for val in opts:
                    if val not in used:
                        used.add(val)
                        assignment[i] = val
                        if backtrack(idx + 1):
                            return True
                        used.remove(val)
                        assignment[i] = None
                return False
            
            if backtrack(0):
                span = max(assignment) - L
                if span < best_span:
                    best_span = span
                    best_result = [int(x) for x in assignment]
                break  # Found minimum R for this L
    
    if best_result is None:
        best_result = divisors[:]
    
    return [int(x) for x in best_result]
