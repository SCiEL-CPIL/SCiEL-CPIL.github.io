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
    timeout = 80  # Safety margin
    
    best_result = None
    best_range = float('inf')
    
    # Strategy: For each possible starting position L (mod LCM),
    # find the tightest window by trying to assign multiples greedily
    # with backtracking for distinctness
    
    for L in range(1, LCM + 1):
        if time.time() - start_time > timeout:
            break
        
        # For each divisor, find multiples in a reasonable range starting from L
        # We'll search for the minimal window
        
        # First, find the earliest multiple >= L for each divisor
        earliest = []
        for d in divisors:
            k = (L + d - 1) // d
            earliest.append(k * d)
        
        # The minimum possible window starts at min(earliest)
        # Let's try windows starting from min(earliest)
        
        # For efficiency, let's try to find assignments for increasing window sizes
        # Start with the natural window [min(earliest), max(earliest)]
        
        base_min = min(earliest)
        base_max = max(earliest)
        
        # Try window sizes from 0 upward, starting from base window
        # But we need to be smarter - let's try multiple starting points
        
        for W in range(0, 200):
            if time.time() - start_time > timeout:
                break
            
            # Try starting positions near L
            # The window is [start, start+W]
            # We need start such that for each divisor, there's a multiple in [start, start+W]
            
            # For divisor d, multiples in [start, start+W] exist iff
            # there's a multiple of d in that range
            # This means: floor((start+W)/d) - ceil(start/d) + 1 >= 1
            
            # Let's try start = L - offset for small offsets
            # Actually, let's just try start = L for this iteration
            
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
            
            # Now find distinct assignment using backtracking
            # Sort by list length (most constrained first)
            indices = list(range(n))
            indices.sort(key=lambda i: len(available[i]))
            
            used = set()
            choices = [0] * n
            
            def backtrack(idx):
                if idx == n:
                    return True
                i = indices[idx]
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
                
                # If we found a good solution, we can break early
                # but let's continue to see if we can do better
                if r == 1:  # Can't do better than 1 (distinct integers)
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
