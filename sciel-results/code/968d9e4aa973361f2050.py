import time
import math
from fractions import Fraction
import numpy as np

def construct():
    deadline = time.time() + 55  # Safety margin
    MAXN = 10_000_000
    TARGET = Fraction(3, 2)
    
    # Initial best
    best_a, best_b = 11, 6
    best_num = abs(2 * best_a - 3 * best_b)
    best_den = 2 * best_b
    
    # Compute sigma(n) for all n <= MAXN
    sigma = np.zeros(MAXN + 1, dtype=np.int64)
    
    for d in range(1, MAXN + 1):
        sigma[d::d] += d
        
    if time.time() > deadline - 10:
        return [int(best_a), int(best_b)]
    
    # Search for pairs
    # a/b close to 3/2 => 2a close to 3b
    max_b = int(MAXN * 2 / 3)
    
    # Window size for a around 1.5b
    # Let's check a in [1.5b - 10, 1.5b + 10]
    # This corresponds to k in [-20, 20]
    
    for b in range(1, max_b + 1):
        if time.time() > deadline - 5:
            break
            
        sb = sigma[b]
        
        # Center a
        a_center = (3 * b) // 2
        
        # Check window
        # We can iterate offset from -10 to 10
        for offset in range(-10, 11):
            a = a_center + offset
            if a < 1 or a > MAXN or a == b:
                continue
                
            if sigma[a] == sb:
                num = abs(2 * a - 3 * b)
                if num == 0:
                    continue
                den = 2 * b
                
                # Compare num/den with best_num/best_den
                # num/den < best_num/best_den
                # num * best_den < best_num * den
                if num * best_den < best_num * den:
                    best_num = num
                    best_den = den
                    best_a = a
                    best_b = b
                    
    return [int(best_a), int(best_b)]
