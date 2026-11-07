import time
from math import gcd
from functools import reduce

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    start_time = time.time()
    deadline = start_time + 85
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, divisors)  # LCM = 30030
    
    best_range = float('inf')
    best_assignment = None
    
    def check_window(s, R):
        """Check if we can assign distinct multiples in [s, s+R] to each divisor."""
        if time.time() > deadline - 2:
            return None
        
        valid = []
        for d in divisors:
            low = max(1, s)
            high = s + R
            if low > high:
                return None
            first = ((low + d - 1) // d) * d
            if first > high:
                return None
            count = (high - first) // d + 1
            valid.append((first, d, count))
        
        # Process in order of increasing count (most constrained first)
        order = sorted(range(n), key=lambda i: valid[i][2])
        used = set()
        assignment = [None] * n
        
        def backtrack(idx):
            if time.time() > deadline - 1:
                return False
            if idx == n:
                return True
            
            i = order[idx]
            first, d, count = valid[i]
            for k in range(count):
                m = first + k * d
                if m not in used:
                    used.add(m)
                    assignment[i] = m
                    if backtrack(idx + 1):
                        return True
                    used.remove(m)
                    assignment[i] = None
            return False
        
        if backtrack(0):
            return assignment[:]
        return None
    
    for R in range(1, 200):
        if time.time() > deadline - 5:
            break
        
        # Generate candidate s values where window boundaries align with multiples
        candidates = set()
        for d in divisors:
            for m in range(d, L + R + 1, d):
                candidates.add((m - R) % L)
                candidates.add(m % L)
        
        for s in sorted(candidates):
            if time.time() > deadline - 3:
                break
            
            result = check_window(s, R)
            if result is not None:
                if len(set(result)) == n and all(isinstance(x, int) and x > 0 and x <= 1000000 for x in result):
                    r = max(result) - min(result)
                    if r < best_range:
                        best_range = r
                        best_assignment = result[:]
                        if R == 1:
                            return [int(x) for x in best_assignment]
                        break
        else:
            continue
        break
    
    if best_assignment is None:
        return [int(d) for d in divisors]
    
    return [int(x) for x in best_assignment]
