import itertools
import time
import random

def check_validity_fast(n, orientation):
    in_nei = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            bit = orientation[idx]
            if bit == 0:
                in_nei[j] |= (1 << i)
            else:
                in_nei[i] |= (1 << j)
            idx += 1
            
    # Precompute pair intersections
    pair_mask = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            m = in_nei[i] & in_nei[j]
            pair_mask[i][j] = m
            pair_mask[j][i] = m
            
    for a in range(n-3):
        for b in range(a+1, n-2):
            mask_ab = pair_mask[a][b]
            if mask_ab == 0:
                return False
            for c in range(b+1, n-1):
                for d in range(c+1, n):
                    if (mask_ab & pair_mask[c][d]) == 0:
                        return False
    return True

def make_paley(p):
    squares = {x*x % p for x in range(1, p)}
    orientation = []
    for i in range(p):
        for j in range(i+1, p):
            if (j-i) % p in squares:
                orientation.append(0)
            else:
                orientation.append(1)
    return orientation

def construct():
    candidates = [31, 43, 47, 59, 67, 83]
    
    for p in candidates:
        orient = make_paley(p)
        if check_validity_fast(p, orient):
            return {'n': p, 'orientation': orient}
            
    # Fallback
    return {'n': 83, 'orientation': make_paley(83)}
