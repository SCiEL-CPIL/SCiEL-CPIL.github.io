    best_n = MIN_N
    best_val = 0
    for n in range(MIN_N, MAX_N):  # n+1 must be <= MAX_N
        if n >= MAX_N:
            break
        prod = pp[n] * pp[n + 1]
        if prod > best_val:
            best_val = prod
            best_n = n
