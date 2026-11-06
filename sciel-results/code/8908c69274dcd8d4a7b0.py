def construct():
    import time
    import numpy as np

    start_time = time.time()
    # Safety margin: The evaluator kills after 90s.
    # We aim to finish in < 5s.
    deadline = start_time + 85.0

    N = 1000001
    # We need lpf for numbers up to 1000001 (since n can be 1000000, n+1 is 1000001)
    # Size N+2 to be safe for indexing
    lpf = np.zeros(N + 2, dtype=np.int32)

    # Sieve for Largest Prime Factor using NumPy
    # We iterate through potential primes.
    # If lpf[i] is 0, i is prime (since no smaller prime has divided it).
    # We then update lpf for all multiples of i.
    # Since we iterate i in increasing order, the last prime factor assigned
    # to any number j will be its largest prime factor.
    
    # We can optimize by only checking i up to N.
    # However, for the sieve to work correctly for "largest" prime factor,
    # we must iterate i from 2 to N.
    # But we can skip even numbers > 2 after handling 2.
    
    # Initialize lpf for even numbers to 2? No, that's not enough for LPF.
    # E.g., 12 = 2^2 * 3. LPF is 3.
    # The standard approach:
    # for i in 2..N:
    #   if lpf[i] == 0: # i is prime
    #     for j in i, 2i, 3i... <= N:
    #       lpf[j] = i
    
    # Vectorized version:
    # for i in range(2, N + 1):
    #   if lpf[i] == 0:
    #     lpf[i::i] = i
    
    # This is still a Python loop over i, but the inner loop is vectorized.
    # Number of primes up to 10^6 is ~78k.
    # Each update lpf[i::i] = i is fast.
    
    # Let's implement this.
    for i in range(2, N + 1):
        if lpf[i] == 0:
            lpf[i::i] = i
        
        # Check time
        if time.time() > deadline:
            # If we run out of time, we must return something.
            # Break and hope we have enough data.
            # But with NumPy, this should be very fast (< 1s).
            break

    # Find n in [10000, 1000000] minimizing max(lpf[n], lpf[n+1])
    # n ranges from 10000 to 1000000 inclusive
    # n+1 ranges from 10001 to 1000001 inclusive
    
    # Create array of n values
    n_vals = np.arange(10000, 1000001)
    
    # Get lpf for n and n+1
    lpf_n = lpf[n_vals]
    lpf_np1 = lpf[n_vals + 1]
    
    # Compute the objective: max of the two
    vals = np.maximum(lpf_n, lpf_np1)
    
    # Find the index of the minimum value
    min_idx = np.argmin(vals)
    best_n = int(n_vals[min_idx])
    
    return best_n
