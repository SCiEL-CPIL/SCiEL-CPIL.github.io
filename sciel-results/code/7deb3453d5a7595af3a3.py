import numpy as np

def construct():
    N = 123457
    lpf = np.ones(N + 1, dtype=np.int32)
    
    # Sieve for Largest Prime Factor
    # We can iterate through potential primes
    # Standard sieve: if lpf[i] == 1, i is prime
    for i in range(2, N + 1):
        if lpf[i] == 1:
            # Mark multiples
            # lpf[i::i] = i  # This vectorizes the assignment
            # But careful: lpf[i::i] creates a view or copy?
            # In NumPy, slice assignment is efficient.
            lpf[i::i] = i
            
    # Calculate max(lpf[a], lpf[N-a]) for all a
    a = np.arange(1, N, dtype=np.int32)
    b = N - a
    vals = np.maximum(lpf[a], lpf[b])
    
    min_val = np.min(vals)
    min_idx = np.argmin(vals)
    
    best_a = int(a[min_idx])
    best_b = N - best_a
    
    return [best_a, best_b]
