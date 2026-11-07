import time
from math import gcd, ceil
from functools import reduce

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
N = len(DIVISORS)
MAX_VAL = 1000000

def construct():
        start_time = time.time()
        deadline = start_time + 80.0
        
        best_span = float('inf')
        best_solution = None
        
        # Scan minimum value m
        # We can scan up to MAX_VAL, but likely a small span is found early.
        # Let's scan up to 1,000,000.
        
        # Optimization: Precompute nothing? Just loop.
        # To speed up, we can check time every 1000 iterations.
        
        for m in range(1, MAX_VAL + 1):
            if time.time() > deadline:
                break
            
            x = [0] * N
            is_distinct = True
            min_x = float('inf')
            max_x = 0
            
            for i in range(N):
                d = DIVISORS[i]
                # Smallest multiple of d >= m
                # k = ceil(m/d)
                # x_i = k * d
                # k = (m + d - 1) // d
                k = (m + d - 1) // d
                val = k * d
                x[i] = val
                if val < min_x: min_x = val
                if val > max_x: max_x = val
            
            # Check distinctness
            # If any collision, skip? Or try to fix?
            # For now, skip if not distinct.
            if len(set(x)) < N:
                continue
                
            span = max_x - min_x
            if span < best_span:
                best_span = span
                best_solution = x[:]
                
                # If span is 0, impossible. Min span for distinct is 1?
                # If span is small, we might break early?
                # But maybe a better one exists?
                # Given the periodicity, the "best" might be global min.
                # Let's keep searching a bit or until deadline.
                
        if best_solution is None:
            # Fallback: just return the divisors themselves?
            # They are distinct?
            # 6, 10, 14, 15, 21, 22, 26, 33, 35, 39 are distinct.
            # Span = 39 - 6 = 33.
            return DIVISORS[:]
            
        return best_solution
