import time

def construct():
    start_time = time.time()
    DEADLINE = 80.0

    N = 123457

    # Sieve for Largest Prime Factor
    # lpf[i] will hold the largest prime factor of i
    lpf = [1] * (N + 1)

    for i in range(2, N + 1):
        if lpf[i] == 1:
            for j in range(i, N + 1, i):
                lpf[j] = i

    best_max_pf = float('inf')
    best_a = 1
    best_b = N - 1

    # Scan all decompositions
    half_N = N // 2
    for a in range(1, half_N + 1):
        if time.time() - start_time > DEADLINE:
            break

        b = N - a
        pa = lpf[a]
        pb = lpf[b]

        m = pa if pa > pb else pb

        if m < best_max_pf:
            best_max_pf = m
            best_a = a
            best_b = b
            if best_max_pf == 2:
                break

    return [best_a, best_b]
