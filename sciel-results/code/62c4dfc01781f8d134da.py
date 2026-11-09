import time
import math
import numpy as np
from scipy.signal import fftconvolve

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Choose L to balance memory and time.
    # L = 50 * 10^6 -> Array size 10^8 -> ~800MB memory.
    L = 50_000_000
    
    # Generate powerful numbers up to L
    powerful = set()
    max_m = int(L ** (1/3)) + 2
    for m in range(1, max_m + 1):
        m3 = m ** 3
        if m3 > L:
            break
        max_k = int(math.isqrt(L // m3))
        for k in range(1, max_k + 1):
            val = k * k * m3
            if val <= L:
                powerful.add(val)
    
    P = sorted(powerful)
    max_val = P[-1] if P else 0
    
    if time.time() > deadline - 50:
        # Fallback to smaller L if generation was slow (unlikely)
        L = 10_000_000
        powerful = set()
        max_m = int(L ** (1/3)) + 2
        for m in range(1, max_m + 1):
            m3 = m ** 3
            if m3 > L:
                break
            max_k = int(math.isqrt(L // m3))
            for k in range(1, max_k + 1):
                val = k * k * m3
                if val <= L:
                    powerful.add(val)
        P = sorted(powerful)
        max_val = P[-1] if P else 0

    # Create array for convolution
    size = 2 * L + 1
    A = np.zeros(size, dtype=np.float64)
    
    for val in P:
        A[val] = 1.0
        
    # Compute convolution
    C = fftconvolve(A, A, mode='full')
    
    if time.time() > deadline - 10:
        # Emergency fallback: return a known small result
        return {'N': 2, 'pairs': [[1, 1]]}

    # Convert to integers
    C_int = np.round(C).astype(np.int64)
    
    # Find N with maximum number of unordered pairs
    max_search = min(size, 2 * max_val + 1)
    
    if max_search <= 2:
        return {'N': 2, 'pairs': [[1, 1]]}

    Ns = np.arange(2, max_search)
    C_vals = C_int[2:max_search]
    
    # Find top candidates
    top_indices = np.argsort(C_vals)[-10:][::-1]
    
    best_N = -1
    best_pairs_count = -1
    
    for idx in top_indices:
        N = Ns[idx]
        raw_count = C_vals[idx]
        
        if N % 2 == 0:
            half = N // 2
            if half in powerful:
                exact_count = (raw_count + 1) // 2
            else:
                exact_count = raw_count // 2
        else:
            exact_count = raw_count // 2
            
        if exact_count > best_pairs_count:
            best_pairs_count = exact_count
            best_N = N
            
    if best_N == -1:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    # Extract pairs for best_N
    pairs = []
    for a in P:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in powerful and a <= b:
            pairs.append([a, b])
            
    return {'N': best_N, 'pairs': pairs}
