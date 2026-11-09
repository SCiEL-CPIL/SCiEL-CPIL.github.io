import time
from math import gcd
from functools import reduce
import bisect
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 90s kill
    
    n = 30
    def lcm(a, b):
        return a // gcd(a, b) * b
    
    L = reduce(lcm, range(1, n + 1))
    
    half = n // 2
    indices1 = range(1, half + 1)
    indices2 = range(half + 1, n + 1)
    
    w1 = [L // i for i in indices1]
    w2 = [L // i for i in indices2]
    
    # Generate sums for half 2
    # Map: sum -> packed_coeffs
    d2 = {0: 0}
    for pos, wi in enumerate(w2):
        new_d2 = {}
        shift = 2 * pos
        for s, p in d2.items():
            # c=-1 (0)
            new_d2[s - wi] = p
            # c=0 (1)
            new_d2[s] = p | (1 << shift)
            # c=1 (2)
            new_d2[s + wi] = p | (2 << shift)
        d2 = new_d2
        
    if time.time() > deadline:
        return [0]*28 + [1, -1]
    
    keys2 = sorted(d2.keys())
    
    best_abs = float('inf')
    best_s1 = 0
    best_s2 = 0
    best_p1 = 0
    best_p2 = 0
    
    # Generate sums for half 1
    d1 = {0: 0}
    for pos, wi in enumerate(w1):
        new_d1 = {}
        shift = 2 * pos
        for s, p in d1.items():
            # c=-1 (0)
            new_d1[s - wi] = p
            # c=0 (1)
            new_d1[s] = p | (1 << shift)
            # c=1 (2)
            new_d1[s + wi] = p | (2 << shift)
        d1 = new_d1
        
        if time.time() > deadline:
            break

    # Iterate d1
    for s1, p1 in d1.items():
        target = -s1
        idx = bisect.bisect_left(keys2, target)
        
        for i in [idx - 1, idx, idx + 1]:
            if 0 <= i < len(keys2):
                s2 = keys2[i]
                total = s1 + s2
                if total != 0:
                    abs_total = abs(total)
                    if abs_total < best_abs:
                        best_abs = abs_total
                        best_s1 = s1
                        best_s2 = s2
                        best_p1 = p1
                        best_p2 = d2[s2]
                        if best_abs == 1:
                            break
        if best_abs == 1:
            break
        if time.time() > deadline:
            break

    if best_abs == float('inf'):
        return [0]*28 + [1, -1]

    # Unpack
    c1 = []
    for i in range(15):
        c1.append(((best_p1 >> (2 * i)) & 3) - 1)
    c2 = []
    for i in range(15):
        c2.append(((best_p2 >> (2 * i)) & 3) - 1)
        
    coeffs = c1 + c2
    
    assert len(coeffs) == 30
    assert all(c in (-1, 0, 1) for c in coeffs)
    
    total_frac = sum(Fraction(c, i + 1) for i, c in enumerate(coeffs))
    assert total_frac != 0
    
    return coeffs
