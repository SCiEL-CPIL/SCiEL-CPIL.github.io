import time
from math import isqrt
import numpy as np

def construct():
    start_time = time.time()
    timeout = 145  # safety margin before 150s limit
    
    # Generate powerful numbers up to M
    # A powerful number is of the form x^2 * y^3 where y is cube-free (or more generally,
    # every prime factor appears with exponent >= 2 in the factorization)
    # The standard generation: p = x^2 * y^3 where y is any integer >= 1
    # But we need to ensure we generate all powerful numbers correctly.
    # A number n is powerful if for every prime p dividing n, p^2 divides n.
    # Every powerful number can be uniquely written as a^2 * b^3 where b is cube-free? No.
    # Actually, every powerful number can be written as x^2 * y^3.
    # Proof: if n = prod p_i^e_i with e_i >= 2, write e_i = 2*a_i + 3*b_i where b_i is 0 or 1?
    # Not quite. But the set {x^2 * y^3 : x,y >= 1} is exactly the set of powerful numbers.
    # Because: if n = x^2 y^3, then for any prime p, the exponent in x^2 y^3 is 2*exp_x(p) + 3*exp_y(p).
    # This is not necessarily >= 2. E.g. x=1, y=2 gives 8=2^3, powerful. x=2, y=1 gives 4=2^2, powerful.
    # x=2, y=2 gives 4*8=32=2^5, powerful.
    # Wait, x=1, y=1 gives 1, powerful.
    # What about x=3, y=1 gives 9, powerful.
    # The key fact: the set of powerful numbers is exactly {x^2 * y^3 : x,y >= 1}.
    # This is a well-known characterization.
    
    M = 200_000_000
    
    powerful = set()
    max_y = int(M ** (1/3)) + 1
    
    for y in range(1, max_y + 1):
        y3 = y ** 3
        if y3 > M:
            break
        max_x = isqrt(M // y3)
        for x in range(1, max_x + 1):
            p = x * x * y3
            if p <= M:
                powerful.add(p)
    
    powerful_list = sorted(powerful)
    powerful_set = set(powerful_list)
    n_list = len(powerful_list)
    
    if n_list < 2:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Use numpy for faster pairwise sum counting
    # We want to count, for each possible sum S, how many pairs (a, b) with a <= b, a+b=S, a,b powerful
    # arr is sorted, so for each i, we can compute arr[i:] + arr[i] and count
    
    arr = np.array(powerful_list, dtype=np.int64)
    max_sum = 2 * M
    counts = np.zeros(max_sum + 1, dtype=np.int32)
    
    # Process in chunks to allow time checking between chunks
    # Use a more efficient approach: for each i, the sums are monotonically increasing
    # We can use np.bincount or direct addition
    
    chunk_size = 2000
    for i in range(0, n_list, chunk_size):
        if time.time() - start_time > timeout - 20:
            break
        end = min(i + chunk_size, n_list)
        for ii in range(i, end):
            # sums = arr[ii] + arr[ii:]
            # These are all >= arr[ii] + arr[ii] = 2*arr[ii]
            # and <= arr[ii] + arr[-1]
            sums = arr[ii] + arr[ii:]
            # Filter to valid range [0, max_sum]
            # Since arr is sorted and non-negative, sums are non-negative
            valid = sums <= max_sum
            if valid.any():
                np.add.at(counts, sums[valid], 1)
    
    if counts.max() == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    best_N = int(np.argmax(counts))
    best_count = int(counts[best_N])
    
    # Reconstruct pairs
    pairs = []
    for a in powerful_list:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in powerful_set:
            pairs.append([a, b])
    
    return {'N': best_N, 'pairs': pairs}
