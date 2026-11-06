import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    N_MIN = 10000
    N_MAX = 1000000
    LIMIT = N_MAX + 1  # We need lpf up to N_MAX+1
    
    # 1. Compute Largest Prime Factor (LPF) using NumPy
    # Initialize lpf array with zeros
    lpf = np.zeros(LIMIT + 1, dtype=np.int32)
    
    # Sieve of Eratosthenes variant: for each prime, update its multiples
    # We iterate up to LIMIT. If lpf[i] is 0, i is prime.
    # Note: In Python, iterating 1 million times is fast enough (<0.1s).
    # The NumPy slice assignment is the key speedup.
    
    # Optimization: We only need to assign if i is prime.
    # We can check lpf[i] == 0.
    
    for i in range(2, LIMIT + 1):
        if lpf[i] == 0:
            # i is prime
            # Assign i to all multiples of i
            # This is a vectorized operation
            lpf[i::i] = i
            
    # Check time after sieve
    if time.time() > deadline:
        # Fallback to a reasonable guess if sieve took too long (unlikely)
        return N_MIN
        
    # 2. Search for n in [N_MIN, N_MAX] that minimizes max(lpf[n], lpf[n+1])
    # Extract the relevant slices
    # n ranges from N_MIN to N_MAX
    # We need lpf[n] for n in [N_MIN, N_MAX]
    # We need lpf[n+1] for n in [N_MIN, N_MAX], which is lpf[N_MIN+1 ... N_MAX+1]
    
    lpf_n = lpf[N_MIN:N_MAX+1]
    lpf_np1 = lpf[N_MIN+1:N_MAX+2]
    
    # Compute max of the two arrays element-wise
    scores = np.maximum(lpf_n, lpf_np1)
    
    # Find the index of the minimum score
    min_idx = np.argmin(scores)
    
    # Convert index to n
    best_n = N_MIN + min_idx
    
    return int(best_n)
