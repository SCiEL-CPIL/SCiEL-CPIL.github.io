def construct():
    import time
    import numpy as np

    start_time = time.time()
    deadline = start_time + 45

    N = 1000000
    limit = N + 1

    # --- Step 1: Sieve of Eratosthenes ---
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i * i::i] = False

    primes = np.where(is_prime)[0]

    # --- Step 2: Compute largest prime factor (LPF) array ---
    lpf = np.zeros(limit + 1, dtype=np.int32)
    for p in primes:
        # Mark all multiples of p with p; since primes are iterated in increasing order,
        # the final value written for each composite is its largest prime factor.
        lpf[p::p] = p

    # --- Step 3: Exact scan for optimal n ---
    # For each n in [10000, 1000000], the objective is max(lpf[n], lpf[n+1])
    lpf_n = lpf[10000:N + 1]
    lpf_np1 = lpf[10001:N + 2]

    # Build score array
    score_arr = np.maximum(lpf_n, lpf_np1)

    # Find global minimum
    min_val = score_arr.min()

    # Among all n achieving the minimum, pick the smallest n (deterministic tie-break)
    candidates = np.where(score_arr == min_val)[0]
    if len(candidates) > 0:
        best_offset = candidates[0]
    else:
        best_offset = 0

    best_n = 10000 + int(best_offset)

    # Safety: ensure within bounds
    if best_n < 10000:
        best_n = 10000
    elif best_n > N:
        best_n = N

    return best_n
