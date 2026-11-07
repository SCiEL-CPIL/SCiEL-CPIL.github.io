import time
import math
from collections import defaultdict

def construct():
    deadline = time.time() + 55.0  # safety margin before 60s
    n = 210
    max_val = 10210
    
    def is_square(m):
        if m < 0:
            return False
        s = math.isqrt(m)
        return s * s == m
    
    # For a*b*210 to be a square, we need a*b = 210*t^2
    # We want to minimize max(a,b), so try small t first
    
    best_max = max_val + 1
    best_result = None
    
    # Iterate t from 1 upward
    t = 1
    while time.time() < deadline:
        M = 210 * t * t
        # a must divide M, a > 210, a < b = M/a, so a < sqrt(M)
        # Also b <= 10210, so a >= M/10210
        # And a <= 10210
        # a > 210
        lo = max(211, (M + 10209) // 10210)
        hi = int(math.isqrt(M))
        if lo > hi:
            t += 1
            continue
        
        found = False
        for a in range(lo, hi + 1):
            if time.time() >= deadline:
                break
            if M % a != 0:
                continue
            b = M // a
            if b > 10210 or b <= a:
                continue
            if b > 210:
                candidate_max = b
                if candidate_max < best_max:
                    best_max = candidate_max
                    best_result = (a, b)
                    found = True
        if found:
            if best_max - 210 <= 10:
                break
        t += 1
        if t > 100000:
            break
    
    # Also try single element: a * 210 = s^2 => a = 210k^2
    # k=2: a=840, check: 840*210 = 176400 = 420^2. Yes!
    
    if best_result is None:
        return [840]
    
    # Verify
    a, b = best_result
    prod = a * b * 210
    if is_square(prod):
        return [int(a), int(b)]
    else:
        return [840]
