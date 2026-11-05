import time
import random
from math import gcd
from bisect import bisect_left
import numpy as np

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin
    
    n = 30
    
    # Calculate L = lcm(1..30)
    def lcm(a, b):
        return a * b // gcd(a, b)
        
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
        
    # A_i = L / i
    A = np.array([L // i for i in range(1, n + 1)], dtype=np.int64)
    
    # Split into two halves
    n1 = 15
    n2 = 15
    
    A1 = A[:n1]
    A2 = A[n1:n1+n2]
    
    # Generate half 1 states using NumPy
    limit1 = 3 ** n1
    # Create array of masks
    masks1 = np.arange(limit1, dtype=np.int32)
    
    # Calculate sums for half 1
    # We need to convert masks to base-3 digits.
    # powers = 3 ** np.arange(n1)
    # digits = (masks1[:, None] // powers[None, :]) % 3
    
    # This creates a 14M x 15 array.
    # To save memory, we can do it in chunks?
    # Or just let it allocate. 14M * 15 * 4 bytes (int32) = 840 MB.
    # This might be too much.
    
    # Let's try a chunked approach or a more memory-efficient way.
    # We can calculate sums iteratively in Python but use NumPy for the heavy lifting?
    # No, let's try to reduce the array size.
    # We can use uint8 for digits?
    # 14M * 15 * 1 byte = 210 MB.
    
    # Let's try to generate digits in chunks to avoid peak memory spike.
    # Or just use the pure Python loop which was estimated at 10-20s.
    # Given the 60s limit, pure Python is safer for memory.
    
    # Let's stick to the pure Python loop for half1 generation but optimize it.
    # We will store (sum, mask) in a list.
    
    half1_data = []
    append1 = half1_data.append
    A1_list = A1.tolist()
    
    for m in range(limit1):
        s = 0
        mm = m
        for j in range(n1):
            d = mm % 3
            mm //= 3
            if d == 1:
                s -= A1_list[j]
            elif d == 2:
                s += A1_list[j]
        append1((s, m))
        
    # Sort half1_data
    half1_data.sort(key=lambda x: x[0])
    
    # Now iterate half2
    limit2 = 3 ** n2
    A2_list = A2.tolist()
    
    best_sum = None
    best_mask1 = None
    best_mask2 = None
    
    # We need to search in half1_data for -s2.
    # half1_data is list of tuples (sum, mask).
    # bisect_left(half1_data, (target, -1)) works because tuple comparison is lexicographical.
    
    for m2 in range(limit2):
        s2 = 0
        mm = m2
        for j in range(n2):
            d = mm % 3
            mm //= 3
            if d == 1:
                s2 -= A2_list[j]
            elif d == 2:
                s2 += A2_list[j]
                
        target = -s2
        
        idx = bisect_left(half1_data, (target, -1))
        
        candidates = []
        if idx < len(half1_data):
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)
            
        for i in candidates:
            s1, mask1 = half1_data[i]
            total_sum = s1 + s2
            
            if total_sum != 0:
                if best_sum is None or abs(total_sum) < abs(best_sum):
                    best_sum = total_sum
                    best_mask1 = mask1
                    best_mask2 = m2
    
    if best_sum is None:
        coeffs = [0] * n
        coeffs[28] = 1
        coeffs[29] = -1
        return coeffs
        
    coeffs = [0] * n
    
    mm = best_mask1
    for j in range(n1):
        d = mm % 3
        mm //= 3
        if d == 1:
            coeffs[j] = -1
        elif d == 2:
            coeffs[j] = 1
    
    mm = best_mask2
    for j in range(n2):
        d = mm % 3
        mm //= 3
        if d == 1:
            coeffs[n1 + j] = -1
        elif d == 2:
            coeffs[n1 + j] = 1
            
    return coeffs
