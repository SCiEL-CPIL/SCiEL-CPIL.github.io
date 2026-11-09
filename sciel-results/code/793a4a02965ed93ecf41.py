import numpy as np
from itertools import product

def construct():
    # Generate the first 20 numbers of the form 2^i * 3^j
    # We can generate a sufficient range and pick the smallest 20
    # Max value 10000.
    # 2^13 = 8192. 3^8 = 6561.
    # i up to 13, j up to 8.
    
    candidates = set()
    for i in range(15):
        for j in range(10):
            val = (2 ** i) * (3 ** j)
            if val <= 10000:
                candidates.add(val)
    
    # Sort and take first 20
    A = sorted(list(candidates))[:20]
    
    # Verify distinct and positive
    if len(A) != 20:
        # Fallback
        return list(range(1, 21))
        
    return A

def score(A):
    sums = set()
    prods = set()
    for a in A:
        for b in A:
            sums.add(a + b)
            prods.add(a * b)
    return max(len(sums), len(prods))
