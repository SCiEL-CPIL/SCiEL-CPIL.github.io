def construct():
    import time
    start = time.time()
    deadline = start + 50  # Leave 40s margin for safety

    N = 1000000
    WINDOW = 24
    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    MAX_VAL = N + WINDOW

    # Precompute smallest prime factor restricted to primes < 24
    lp = [0] * (MAX_VAL + 1)
    for p in PRIMES:
        for m in range(p, MAX_VAL + 1, p):
            if lp[m] == 0:
                lp[m] = p

    # Compute sp[m] = complete factor of m composed of primes < 24
    sp = [1] * (MAX_VAL + 1)
    for m in range(2, MAX_VAL + 1):
        p = lp[m]
        if p:
            sp[m] = sp[m // p] * p

    best_n = 1000
    best_distinct = WINDOW + 1

    # Sliding window with optimized local references
    freq = {}
    current_distinct = 0
    sp_local = sp

    # Initialize window for n=1000: sp[1001] ... sp[1024]
    for i in range(1001, 1001 + WINDOW):
        val = sp_local[i]
        count = freq.get(val, 0)
        if count == 0:
            current_distinct += 1
        freq[val] = count + 1

    if current_distinct < best_distinct:
        best_distinct = current_distinct
        best_n = 1000

    # Main sliding window scan
    for n in range(1001, N + 1):
        # Check time periodically to avoid overhead
        if (n & 1023) == 0 and time.time() > deadline:
            break

        # Remove sp[n]
        old_val = sp_local[n]
        count = freq[old_val]
        if count == 1:
            del freq[old_val]
            current_distinct -= 1
        else:
            freq[old_val] = count - 1

        # Add sp[n+WINDOW]
        new_val = sp_local[n + WINDOW]
        count = freq.get(new_val, 0)
        if count == 0:
            current_distinct += 1
        freq[new_val] = count + 1

        if current_distinct < best_distinct:
            best_distinct = current_distinct
            best_n = n
            if best_distinct <= 2:
                break

    return best_n
