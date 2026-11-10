import time
from math import gcd
from functools import reduce

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    # LCM of divisors
    def lcm(a, b):
        return a * b // gcd(a, b)
    LCM = reduce(lcm, divisors)  # 30030
    
    start_time = time.time()
    timeout = 85  # Safety margin for 90s limit
    
    best_result = None
    best_range = float('inf')
    
    # Pre-sort indices by divisor size descending (larger divisors are more constrained)
    sorted_indices = sorted(range(n), key=lambda i: -divisors[i])
    
    for L in range(1, LCM + 1):
        elapsed = time.time() - start_time
        if elapsed > timeout:
            break
        
        # Find the earliest multiple >= L for each divisor
        earliest = []
        for d in divisors:
            k = (L + d - 1) // d
            earliest.append(k * d)
        
        base_min = min(earliest)
        base_max = max(earliest)
        min_W = base_max - base_min
        
        # Try window sizes from min_W upward (we know at least one multiple exists per divisor)
        for W in range(min_W, min_W + 150):
            elapsed = time.time() - start_time
            if elapsed > timeout:
                break
            
            start = L
            end = start + W
            
            # Find available multiples for each divisor in [start, end]
            available = []
            valid = True
            for d in divisors:
                k = (start + d - 1) // d
                first = k * d
                if first > end:
                    valid = False
                    break
                multiples = []
                x = first
                while x <= end:
                    multiples.append(x)
                    x += d
                available.append(multiples)
            
            if not valid:
                continue
            
            # Backtracking with pre-sorted order (most constrained first)
            used = set()
            choices = [0] * n
            
            def backtrack(idx):
                if idx == n:
                    return True
                i = sorted_indices[idx]
                for val in available[i]:
                    if val not in used:
                        used.add(val)
                        choices[i] = val
                        if backtrack(idx + 1):
                            return True
                        used.remove(val)
                return False
            
            if backtrack(0):
                result = [choices[i] for i in range(n)]
                r = max(result) - min(result)
                if r < best_range:
                    best_range = r
                    best_result = result[:]
                
                if r == 1:
                    return best_result
    
    if best_result is not None:
        # Verify
        assert len(set(best_result)) == n
        for i in range(n):
            assert best_result[i] % divisors[i] == 0
            assert 0 < best_result[i] <= 1000000
        return best_result
    
    # Fallback
    return divisors[:]
