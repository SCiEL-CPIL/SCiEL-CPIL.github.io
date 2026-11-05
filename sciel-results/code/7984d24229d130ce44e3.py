import time
from math import isqrt
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    def time_left():
        return deadline - time.time()
        
    # Try increasing M
    # M=10^8 is safe. M=2*10^8 might be tight.
    M = 200_000_000
    
    # Generate powerfuls
    max_b = int(M ** (1.0 / 3.0)) + 2
    
    def is_square_free(n):
        if n < 2: return n == 1
        temp = n
        p = 2
        while p * p <= temp:
            if temp % (p * p) == 0: return False
            while temp % p == 0: temp //= p
            p += 1
        return True

    powerful = set()
    for b in range(1, max_b + 1):
        if not is_square_free(b): continue
        b_cubed = b ** 3
        if b_cubed > M: break
        max_a = isqrt(M // b_cubed)
        for a in range(1, max_a + 1):
            powerful.add(a * a * b_cubed)
            
    if time.time() > deadline - 20:
        # Fallback
        return {'N': 2, 'pairs': [[1, 1]]}

    # FFT
    # Size must be at least 2*M
    # Use next power of 2 for FFT speed
    size = 1
    while size < 2 * M:
        size *= 2
        
    # Check memory: size * 4 bytes (float32)
    # If size = 2^28 = 268M, mem = 1GB. OK.
    # If M=2e8, 2M=4e8. Next power 2^29 = 536M. Mem 2GB. OK.
    
    A = np.zeros(size, dtype=np.float32)
    for p in powerful:
        A[p] = 1.0
        
    # FFT
    B = np.fft.rfft(A)
    C = B * B
    D = np.fft.irfft(C, n=size)
    
    # D contains the counts
    # Max count
    max_count = np.max(D)
    best_N = np.argmax(D)
    
    # Reconstruct pairs
    pairs = []
    P_set = powerful
    for a in powerful:
        if 2 * a > best_N:
            break
        b = best_N - a
        if b in P_set and a <= b:
            pairs.append([a, b])
            
    return {'N': int(best_N), 'pairs': pairs}
