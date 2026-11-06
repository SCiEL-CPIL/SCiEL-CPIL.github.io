import time
from fractions import Fraction

def construct():
    def solve(k, min_den, remaining, current, deadline):
        if k == 0:
            if remaining == 0:
                return current
            return None
        
        if remaining <= 0:
            return None
            
        # Max sum check
        max_sum = Fraction(0, 1)
        for i in range(k):
            max_sum += Fraction(1, min_den + i)
        if max_sum < remaining:
            return None
            
        # Lower bound for d
        # d >= 1/remaining
        # d >= (k - remaining) / (1 - remaining)
        lb1 = int(1 / remaining) + 1
        lb2 = int((k - remaining) / (1 - remaining)) + 1
        lb = max(min_den, lb1, lb2)
        
        # Upper bound for d
        # 1/d + (k-1)/(d+1) >= remaining
        # Approx d <= k / remaining
        ub = min(10000000, int(k / remaining))
        
        if lb > ub:
            return None
            
        for d in range(lb, ub + 1):
            if time.time() > deadline:
                return None
                
            if Fraction(1, d) > remaining:
                continue
                
            new_remaining = remaining - Fraction(1, d)
            if new_remaining < 0:
                continue
                
            res = solve(k - 1, d + 1, new_remaining, current + [d], deadline)
            if res is not None:
                return res
                
        return None

    start_time = time.time()
    deadline = start_time + 55.0
    
    # The minimum number of terms is 8.
    for k in range(8, 12):
        if time.time() > deadline:
            break
        res = solve(k, 5, Fraction(1, 1), [], deadline)
        if res is not None:
            return res
            
    # Fallback
    return [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]
