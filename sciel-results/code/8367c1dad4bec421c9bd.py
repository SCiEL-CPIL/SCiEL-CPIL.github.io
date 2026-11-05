import time
from math import isqrt
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin (killed at 150s)
    
    # Generate powerful numbers up to M
    # A powerful number is n = u^2 * v^3 where v is square-free
    M = 80_000_000
    
    powerful_set = set()
    max_v = int(M ** (1/3)) + 2
    
    def is_square_free(n):
        if n < 1: return False
        if n == 1: return True
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
            if p * p > n:
                break
            if n % (p * p) == 0:
                return False
        return True

    for v in range(1, max_v + 1):
        if not is_square_free(v):
            continue
        v3 = v ** 3
        if v3 > M:
            break
        max_u = isqrt(M // v3)
        for u in range(1, max_u + 1):
            powerful_set.add(u * u * v3)
            
    P = sorted(powerful_set)
    K = len(P)
    
    if time.time() > deadline - 30:
        # Fallback: just return a small valid answer
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Use FFT convolution to count representations
    # Create indicator array
    arr = np.zeros(2 * M + 1, dtype=np.float32)
    for x in P:
        arr[x] = 1.0
    
    # FFT size: next power of 2 >= 4*M+1
    n_fft = 1
    while n_fft < 4 * M + 1:
        n_fft *= 2
    
    # Convolve
    fft = np.fft.fft(arr, n=n_fft)
    conv = np.fft.ifft(fft * fft).real
    
    # conv[n] = number of ordered pairs (a,b) with a+b=n, a,b in P
    # For a<=b pairs: count = (conv[n] + (1 if n%2==0 and n//2 in P else 0)) / 2
    # Actually: unordered pairs with a<=b:
    #   Let c(n) = #{(a,b) in P x P : a+b=n} = conv[n]
    #   Let d(n) = #{a in P : 2a=n} = 1 if n even and n/2 in P else 0
    #   Pairs with a<=b: (c(n) + d(n)) / 2
    
    # Find max
    # conv indices 0..2M
    # We want n >= 2 (since N >= 2)
    max_count = 0
    best_N = 2
    P_set = set(P)
    
    for n in range(2, 2 * M + 1):
        c = int(round(conv[n]))
        d = 1 if (n % 2 == 0 and n // 2 in P_set) else 0
        pairs_count = (c + d) // 2
        if pairs_count > max_count:
            max_count = pairs_count
            best_N = n
    
    if time.time() > deadline - 10:
        # Still need to reconstruct pairs, but be quick
        pass
    
    # Reconstruct pairs for best_N
    pairs = []
    for a in P:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in P_set:
            pairs.append([a, b])
    
    return {'N': best_N, 'pairs': pairs}
