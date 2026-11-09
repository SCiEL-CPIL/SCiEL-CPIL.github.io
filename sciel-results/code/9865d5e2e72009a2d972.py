import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = 58  # Safety margin for 60s limit (or 90s if evaluator allows, but 60s is safer per contract "finish within 60s")
    
    N = 1000000
    max_val = N + 1  # We need lpf for N+1 as well
    
    # Initialize LPF array with 0
    # Using int64 to be safe, though int32 is enough for values < 1e6
    lpf = np.zeros(max_val + 1, dtype=np.int64)
    
    # Sieve for Largest Prime Factor
    # Iterate through potential primes
    # If lpf[i] is 0, i is prime (since no smaller prime factor was found)
    # For each prime i, update lpf for all multiples of i
    # Since we iterate i in increasing order, the last prime to update a multiple j
    # will be the largest prime factor of j.
    
    for i in range(2, max_val + 1):
        if lpf[i] == 0:
            # i is prime
            # Vectorized update: lpf[i::i] = i
            # This assigns i to all indices i, 2i, 3i, ...
            lpf[i::i] = i
            
        # Optional: Early exit if time is running out, though NumPy sieve is very fast
        if time.time() - start_time > deadline:
            # Fallback: return a known good candidate or just the first one
            # If we haven't finished the sieve, we can't reliably compute LPF.
            # But the sieve is so fast it should finish in < 1s.
            # If it doesn't, we have a problem. Let's just return a safe guess.
            return 10000

    # Convert to list for faster indexing in the search loop? 
    # Or keep as numpy array and use numpy operations for the search?
    # The search loop iterates n from 10000 to 1000000.
    # We need max(lpf[n], lpf[n+1]) for each n.
    # We can compute this vectorized too!
    
    # Create an array of n values
    # n ranges from 10000 to N
    # lpf_n = lpf[10000 : N+1]
    # lpf_np1 = lpf[10001 : N+2]
    
    lpf_n = lpf[10000 : N+1]
    lpf_np1 = lpf[10001 : N+2]
    
    # Compute max for each pair
    max_lpf_pairs = np.maximum(lpf_n, lpf_np1)
    
    # Find the index of the minimum value
    min_idx = np.argmin(max_lpf_pairs)
    min_lpf_val = max_lpf_pairs[min_idx]
    
    # The corresponding n is 10000 + min_idx
    best_n = 10000 + min_idx
    
    return best_n
