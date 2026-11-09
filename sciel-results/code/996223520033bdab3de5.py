"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    n = 30
    def lcm(a, b):
        return a * b // gcd(a, b)
    L = reduce(lcm, range(1, n + 1))
    Li = [L // i for i in range(1, n + 1)]
    
    def numerator(coeffs):
        s = 0
        for i, c in enumerate(coeffs):
            if c != 0:
                s += c * Li[i]
        return s
    
    def abs_nonzero(coeffs):
        s = numerator(coeffs)
        if s == 0:
            return float('inf')
        return abs(s)
    
    # Start from seed
    coeffs = [0] * 28 + [1, -1]
    best_coeffs = coeffs[:]
    best_abs = abs_nonzero(coeffs)
    
    random.seed(12345)
    
    # Local search: hill climbing with random restarts
    while time.time() < deadline - 5:
        # Hill climb from current position
        improved = True
        while improved:
            improved = False
            # Try all single-coordinate changes
            for i in range(n):
                if time.time() > deadline - 5:
                    break
                old = coeffs[i]
                for new_val in [-1, 0, 1]:
                    if new_val == old:
                        continue
                    coeffs[i] = new_val
                    new_abs = abs_nonzero(coeffs)
                    if new_abs < best_abs:
                        best_abs = new_abs
                        best_coeffs = coeffs[:]
                        improved = True
                        break
                    else:
                        coeffs[i] = old
            if time.time() > deadline - 5:
                break
        
        # Random restart
        if time.time() > deadline - 5:
            break
        coeffs = [random.choice([-1, 0, 1]) for _ in range(n)]
        cur_abs = abs_nonzero(coeffs)
        if cur_abs < best_abs:
            best_abs = cur_abs
            best_coeffs = coeffs[:]
    
    return best_coeffs
# EVOLVE-BLOCK-END
