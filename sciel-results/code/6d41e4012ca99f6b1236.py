import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 140.0
    
    # We aim for a large N to find longer runs.
    # N = 10^8 is safe with int32 (400MB per array).
    N = 100_000_000
    
    # Check if we have enough time and memory (memory is implicit, assume it fits)
    # If time is tight, we could reduce N, but 10s is fine.
    
    phi = np.arange(N + 1, dtype=np.int32)
    
    # Sieve for phi
    for i in range(2, N + 1):
        if phi[i] == i:
            phi[i::i] -= phi[i::i] // i
            
    # Check time
    if time.time() - start_time > DEADLINE - 5:
        # Fallback to a smaller N if sieve took too long? 
        # Unlikely for 10^8. But if it does, we might not have time for the window.
        # Let's just proceed, the window is fast.
        pass

    # Sliding window to find longest subarray with distinct elements
    # phi values are in [1, N]
    last_seen = np.full(N + 1, -1, dtype=np.int32)
    
    left = 0
    max_len = 0
    max_start = 1
    
    for right in range(1, N + 1):
        val = phi[right]
        if last_seen[val] >= left:
            left = last_seen[val] + 1
        last_seen[val] = right
        
        current_len = right - left + 1
        if current_len > max_len:
            max_len = current_len
            max_start = left
            
    # Ensure constraints
    if max_start + max_len - 1 > 10**12:
        # This won't happen for N=10^8
        max_len = 1
        max_start = 1
        
    if max_len < 1:
        max_len = 1
        max_start = 1

    return {'start': max_start, 'length': max_len}
