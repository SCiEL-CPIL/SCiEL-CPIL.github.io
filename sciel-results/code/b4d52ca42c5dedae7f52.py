import time
import numpy as np
from math import isqrt

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit
    
    # Target limit for powerful numbers
    # We can afford a large limit due to FFT efficiency
    L = 20_000_000  # 2 * 10^7
    
    # Check if we have time to generate
    if time.time() - start_time > deadline:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    # Generate powerful numbers up to L
    # Powerful numbers are of the form a^2 * b^3 where a is square-free
    # We can iterate b (cube part) and a (square part)
    # b^3 <= L => b <= L^(1/3)
    # a^2 <= L/b^3 => a <= sqrt(L/b^3)
    
    powerful_set = set()
    max_b = isqrt(isqrt(L))  # Actually L^(1/3)
    # L^(1/3) for 2*10^7 is approx 271
    max_b = int(L ** (1/3)) + 1
    
    for b in range(1, max_b + 1):
        b3 = b ** 3
        if b3 > L:
            break
        max_a = isqrt(L // b3)
        for a in range(1, max_a + 1):
            # Check if a is square-free?
            # Actually, the representation a^2 b^3 is unique if a is square-free.
            # If a is not square-free, say a = c^2 * d, then a^2 b^3 = c^4 d^2 b^3 = (c^2 b)^2 d^2 ...
            # Standard generation: iterate square-free a.
            # However, simply generating all a and b and taking the set handles duplicates automatically.
            val = a * a * b3
            if val <= L:
                powerful_set.add(val)
            else:
                break # Since a increases, val increases
                
    # Add 1 explicitly if not covered (1 = 1^2 * 1^3, so it is covered)
    if 1 not in powerful_set:
        powerful_set.add(1)
        
    # Convert to array for FFT
    # We need an array of size at least 2*L + 1
    arr_size = 2 * L + 1
    # Find next power of 2 for FFT efficiency
    n_fft = 1
    while n_fft < arr_size:
        n_fft *= 2
        
    A = np.zeros(n_fft, dtype=np.float64)
    for p in powerful_set:
        A[p] = 1.0
        
    # Check time
    if time.time() - start_time > deadline:
        # Fallback to simple search if FFT takes too long to setup? 
        # But FFT is fast. If we are here, we are ready.
        pass

    # Perform FFT
    F = np.fft.fft(A)
    C = np.fft.ifft(F * F)
    C = np.real(C)
    
    # Find the index with max count
    # We only care about indices up to 2*L
    # Also N must be >= 2
    max_count = 0
    best_N = 2
    # Search from 2 to 2*L
    # Using argmax on the slice
    counts = C[2:2*L+1]
    if len(counts) > 0:
        best_idx = np.argmax(counts)
        max_count = counts[best_idx]
        best_N = best_idx + 2
        
    # If max_count is 0, something went wrong, but 1+1=2 should be there.
    
    # Check time before reconstruction
    if time.time() - start_time > deadline:
        return {'N': best_N, 'pairs': []} # Invalid pairs? No, we need valid pairs.
        # If we run out of time, we should return a valid small result.
        # Let's ensure we have at least one pair.
        # 1 is powerful. If best_N is large, we might not have time to find pairs.
        # But reconstruction is O(num_powerful). ~30k iterations. Very fast.
        
    # Reconstruct pairs
    powerful_list = sorted(powerful_set)
    pairs = []
    # We need a <= b, so a <= N/2
    for a in powerful_list:
        if a * 2 > best_N:
            break
        b = best_N - a
        if b in powerful_set:
            pairs.append([a, b])
            
    return {'N': best_N, 'pairs': pairs}
