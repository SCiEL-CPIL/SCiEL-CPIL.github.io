import numpy as np
from scipy.signal import fftconvolve
import time
from math import isqrt

def construct():
    start_time = time.time()
    # Safety margin: return before 110s
    deadline = start_time + 110.0
    
    # We want to maximize the number of representations.
    # Search space: Powerful numbers up to L.
    # Larger L -> more powerful numbers -> potentially more collisions.
    # Memory constraint: Array of size 2*L.
    # L=1e8 -> 2e8 size. 800MB (int32).
    # L=2e8 -> 4e8 size. 1.6GB.
    # Let's try L=2e8 if memory allows, else 1e8.
    
    # We'll try to estimate if 2e8 is feasible.
    # For now, start with 1e8 which is very safe.
    # If we have time, we could try larger, but convolution is the bottleneck.
    
    L = 10**8
    
    # Generate powerful numbers
    powerful = set()
    # v^3 <= L
    max_v = int(L ** (1.0/3.0)) + 2
    for v in range(1, max_v + 1):
        v3 = v ** 3
        if v3 > L:
            break
        # u^2 * v^3 <= L => u <= sqrt(L / v^3)
        max_u = isqrt(L // v3)
        for u in range(1, max_u + 1):
            p = u * u * v3
            powerful.add(p)
            
    P = sorted(powerful)
    # P contains 1, 4, 8, 9, ...
    
    # Create indicator vector
    # Size L+1
    ind = np.zeros(L + 1, dtype=np.int32)
    for p in P:
        ind[p] = 1
        
    # Check time
    if time.time() > deadline - 30:
         # Fallback: just pick a small N?
         # Unlikely to happen with L=1e8 generation
         pass

    # Compute convolution
    # fftconvolve is faster for large arrays
    # mode='full' gives size 2*L + 1
    try:
        counts = fftconvolve(ind, ind, mode='full')
    except Exception:
        # Fallback to numpy convolve if scipy fails or is slow?
        # Or just use smaller L
        counts = np.convolve(ind, ind, mode='full')
        
    # Find N with max count
    # N must be >= 2
    # counts[i] is the number of pairs (a,b) with a+b=i
    # We want to maximize counts[i]
    
    # Ensure we don't pick index 0 or 1 (counts are 0 anyway)
    # Find argmax
    # Note: counts is float64 from fftconvolve. Cast to int for safety?
    # The values are small enough to fit in int.
    
    max_idx = np.argmax(counts)
    N = int(max_idx)
    
    # If N is too small or something weird, check neighbors?
    # No, argmax is fine.
    
    # Extract pairs
    P_set = set(P)
    pairs = []
    for a in P:
        if a > N:
            break
        b = N - a
        if b < a:
            continue
        if b in P_set:
            pairs.append([a, b])
            
    # Verify N <= 10^18
    if N > 10**18:
        # This shouldn't happen with L=1e8
        return {'N': 2, 'pairs': [[1, 1]]}
        
    return {'N': N, 'pairs': pairs}
