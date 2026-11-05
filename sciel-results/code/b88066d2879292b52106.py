import time
import numpy as np

def construct():
    # Deadline guard, though this method is very fast
    deadline = time.time() + 85.0
    
    # We need LPF for numbers up to 1,000,001 (since n can be 1,000,000, n+1 is 1,000,001)
    MAXV = 1000001
    
    # Initialize LPF array with 0s. 0 will indicate "not yet visited" or "prime itself" initially.
    # We use int64 to be safe, though int32 would suffice for values up to 1e6.
    lpf = np.zeros(MAXV + 2, dtype=np.int32)
    
    # Sieve for Largest Prime Factor
    # Iterate p from 2 to MAXV.
    # If lpf[p] is 0, p is prime.
    # For a prime p, all multiples of p have p as a prime factor.
    # Since we iterate p in increasing order, the last prime p that visits a number x
    # is the largest prime factor of x.
    for p in range(2, MAXV + 1):
        if lpf[p] == 0:
            # p is prime
            # Mark all multiples of p with p.
            # This slice assignment is vectorized and fast.
            lpf[p::p] = p
            
    if time.time() > deadline:
        # Fallback if sieve somehow took too long (highly unlikely)
        return 10000

    # We need to check n in [10000, 1000000].
    # So we need lpf[n] for n in [10000, 1000000]
    # and lpf[n+1] for n in [10000, 1000000], which means lpf indices up to 1000001.
    
    # Extract the relevant slices
    # n ranges from 10000 to 1000000 inclusive.
    # lpf_n corresponds to lpf[10000] ... lpf[1000000]
    # lpf_np1 corresponds to lpf[10001] ... lpf[1000001]
    
    lpf_n = lpf[10000:1000001]
    lpf_np1 = lpf[10001:1000002]
    
    # Calculate the objective: max(LPF(n), LPF(n+1))
    max_lpf = np.maximum(lpf_n, lpf_np1)
    
    # Find the index of the minimum value
    min_idx = np.argmin(max_lpf)
    
    best_n = 10000 + min_idx
    
    return int(best_n)
