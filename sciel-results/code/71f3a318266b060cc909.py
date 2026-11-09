def construct():
    import time
    start = time.time()
    deadline = start + 85

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
        if time.time() > deadline - 5:  # reserve 5s for secondary pass
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

    # Secondary targeted search: coarse scan to find promising regions, then fine scan
    if time.time() < deadline - 2 and best_distinct > 3:
        # Coarse scan with step 500
        step = 500
        for n in range(1000, N + 1, step):
            if time.time() > deadline - 2:
                break
            # Quick evaluation of this window
            factors = set()
            for i in range(1, WINDOW + 1):
                factors.add(sp_local[n + i])
            val = len(factors)
            if val < best_distinct:
                best_distinct = val
                best_n = n

        # Fine scan around best candidate
        if time.time() < deadline - 1:
            lo = max(1000, best_n - 200)
            hi = min(N, best_n + 200)
            for n in range(lo, hi + 1):
                if time.time() > deadline - 1:
                    break
                factors = set()
                for i in range(1, WINDOW + 1):
                    factors.add(sp_local[n + i])
                val = len(factors)
                if val < best_distinct:
                    best_distinct = val
                    best_n = n

    return best_n
