import time
import sys

# EVOLVE-BLOCK-START
P = {'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39], 'maximum': 1000000}

def construct():
    divisors = P['divisors']
    max_val = P['maximum']
    
    best_range = float('inf')
    best_xs = None
    best_m = None
    
    # We iterate m from 1 to max_val.
    # The function max(xs) is non-decreasing, so we can break early.
    # 1,000,000 iterations * 10 divisors is ~10^7 operations, which is fast in Python.
    
    for m in range(1, max_val + 1):
        # Calculate the smallest multiple of d >= m for each d
        # d * ceil(m/d)
        xs = [d * ((m + d - 1) // d) for d in divisors]
        
        mx = max(xs)
        
        # If the maximum exceeds the limit, further m will only increase mx (non-decreasing)
        if mx > max_val:
            break
            
        mn = min(xs)
        current_range = mx - mn
        
        if current_range < best_range:
            best_range = current_range
            best_xs = xs
            best_m = m
            
    # If no solution found (should not happen), return divisors as fallback
    if best_xs is None:
        return P['divisors'][:]
        
    return best_xs
# EVOLVE-BLOCK-END
