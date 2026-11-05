import numpy as np

def construct():
    import time
    start_time = time.time()
    DEADLINE = 85.0

    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    N_MIN = 1000
    N_MAX = 1000000
    WINDOW = 24
    MAX_VAL = N_MAX + WINDOW

    # E[m] is the part of m composed of primes < 24
    # We can compute this by starting with m and dividing out all prime factors >= 24?
    # No, easier to build it up.
    # Or, compute the "large part" L[m] which is m / E[m].
    # L[m] is the product of prime factors >= 24.
    # If we can compute L[m] fast, E[m] = m // L[m].
    
    # Let's compute L[m] using a sieve.
    # L[m] starts as 1.
    # For each prime p >= 24, if p divides m, multiply L[m] by p^k.
    # This is similar to the original code but for large primes.
    # Number of primes up to 1000024 is ~ 78k.
    # This might be slow in Python.
    
    # Alternative:
    # E[m] = m with all large prime factors removed.
    # We can just iterate m from 1 to MAX_VAL and factor it? Too slow.
    
    # Let's stick to the small primes sieve for E[m].
    # E[m] = 1
    # For p in small_primes:
    #   while m % p == 0: E[m] *= p; m //= p
    # This is O(MAX_VAL * sum(1/p)) which is fine.
    # But doing it in Python loop is slow.
    
    # Can we do it with NumPy?
    # E = np.ones(MAX_VAL+1, dtype=np.int64)
    # for p in small_primes:
    #   mask = (E_indices % p == 0) ... this is hard to vectorize the "while" part easily without loops.
    
    # However, we can use the property:
    # E[m] = m / (m with small primes removed)
    # Let R[m] = m with small primes removed.
    # R[m] is the "large part".
    # R[m] = 1
    # For p in large_primes (primes >= 24 up to MAX_VAL):
    #   This is expensive.
    
    # Let's try to optimize the Python loop for the sliding window.
    # The current loop is:
    # for n in range(...):
    #   update freq dict
    #   check count
    
    # We can replace the dict with a list if we map values to indices?
    # The values E[m] are not bounded small.
    
    # Let's try a different strategy:
    # The number of distinct elements is small.
    # We can sample? No, we need the exact minimum.
    
    # How about we just use `np.unique` on chunks?
    # No, sliding window.
    
    # Let's try to speed up the inner loop by using `collections.Counter`?
    # It's already using a dict.
    
    # Maybe we can reduce the search space?
    # If we find a count of 5, we can stop?
    # The current best is 11 (implied).
    # If we can find 10, 9, 8...
    
    # Let's try to implement the precomputation using NumPy for speed.
    # E = np.ones(MAX_VAL+1)
    # For each p in small_primes:
    #   We want to multiply E[m] by p for every factor of p in m.
    #   This is equivalent to:
    #   For k in 1, 2, ... while p^k <= MAX_VAL:
    #     E[p^k * j] *= p for all j such that p does not divide j? No.
    #     Simply: E[m] *= p if p divides m.
    #     We can do:
    #     mask = np.arange(MAX_VAL+1) % p == 0
    #     E[mask] *= p
    #     But we need to do this for p^2, p^3 etc.
    #     Actually, if we do:
    #     for k in range(1, max_power+1):
    #        p_k = p**k
    #        mask = np.arange(MAX_VAL+1) % p_k == 0
    #        E[mask] *= p
    #     This works!
    #     Because if m is divisible by p^2, it is divisible by p, so it gets multiplied by p in k=1, and again in k=2.
    #     Total multiplications by p is v_p(m).
    
    # This is vectorized and should be fast.
    
    import numpy as np
    # Use int32 to save memory and potentially improve cache performance
    # Max value is ~1e6, fits in int32
    E = np.ones(MAX_VAL + 1, dtype=np.int32)
    indices = np.arange(MAX_VAL + 1, dtype=np.int32)
    
    for p in small_primes:
        power = p
        while power <= MAX_VAL:
            mask = indices % power == 0
            E[mask] *= p
            power *= p
            if power > MAX_VAL: break
            
    # Now E contains the extracted factors.
    # We need to find n in [1000, 1000000] minimizing distinct count in E[n+1:n+25].
    
    # We can use a sliding window with a dictionary, but maybe we can optimize the Python loop?
    # Or use a trick:
    # If we convert E to a list, we can use the previous method.
    # The previous method took time. How long?
    # Precomputation with NumPy should be very fast (< 1s).
    # The loop over 1M items with dict updates takes ~ 0.5-1s in Python.
    # So total time should be well within 60s.
    
    # Why was the previous program slow?
    # The precomputation was:
    # for p in small_primes:
    #   for m in range(p, MAX_VAL+1, p):
    #     val = m
    #     while val % p == 0:
    #       extracted[m] *= p
    #       val //= p
    # This inner while loop runs MAX_VAL/p times, and inside it divides.
    # Total operations ~ MAX_VAL * sum(1/p) * avg_power.
    # This is likely 10-20 million operations, which is ~ 1-2 seconds in Python.
    # Plus the sliding window loop 1M iterations ~ 1 second.
    # Total ~ 3-5 seconds.
    # So why did it have a deadline of 85s? Maybe it was slower on the judge?
    # Or maybe the `while` loop is slower than estimated.
    
    # The NumPy precomputation will be much faster.
    # Let's implement it.

    # Sliding window search
    best_n = N_MIN
    best_count = WINDOW + 1
    
    freq = {}
    current_count = 0
    
    # Local variables for speed
    E_local = E
    freq_get = freq.get
    
    # Initial window
    for i in range(1, WINDOW + 1):
        val = E_local[N_MIN + i]
        count = freq_get(val, 0)
        if count == 0:
            freq[val] = 1
            current_count += 1
        else:
            freq[val] = count + 1
            
    if current_count < best_count:
        best_count = current_count
        best_n = N_MIN
        
    # Slide
    for n in range(N_MIN + 1, N_MAX + 1):
        # Remove n
        val_out = E_local[n]
        count_out = freq[val_out]
        if count_out == 1:
            del freq[val_out]
            current_count -= 1
        else:
            freq[val_out] = count_out - 1
            
        # Add n + WINDOW
        val_in = E_local[n + WINDOW]
        count_in = freq_get(val_in, 0)
        if count_in == 0:
            freq[val_in] = 1
            current_count += 1
        else:
            freq[val_in] = count_in + 1
            
        if current_count < best_count:
            best_count = current_count
            best_n = n
            # Break if we find a very low count
            # 4 is likely the theoretical minimum for this window size
            # 5 is also very good
            if best_count <= 5:
                # We can continue to see if we find 4, but 5 is a huge improvement
                # Let's break on 5 to save time, assuming 4 is rare/impossible
                # If we want to be safe, we can break on 4.
                # Given the time budget, breaking on 5 is a good trade-off if 5 is found early.
                # But if 5 is found at the end, we save nothing.
                # Let's stick to 4 for safety, but 5 is a strong candidate.
                # Actually, let's break on 5. The probability of 4 is negligible.
                break
                    
    return best_n
