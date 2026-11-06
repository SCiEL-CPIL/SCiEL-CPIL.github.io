import time

def construct():
    deadline = time.time() + 85
    TARGET = 123457

    # Sieve to compute largest prime factor for all integers up to TARGET
    lpf = [0] * (TARGET + 1)
    lpf[1] = 1
    for p in range(2, TARGET + 1):
        if lpf[p] == 0:  # p is prime
            for j in range(p, TARGET + 1, p):
                lpf[j] = p

    # Linear scan: find a in [1, TARGET-1] that minimizes max(lpf[a], lpf[TARGET-a])
    best_val = float('inf')
    best_a = 1
    best_b = TARGET - 1
    for a in range(1, TARGET):
        if time.time() > deadline - 5:
            break
        b = TARGET - a
        pa = lpf[a]
        pb = lpf[b]
        m = pa if pa > pb else pb
        if m < best_val:
            best_val = m
            best_a = a
            best_b = b

    return [best_a, best_b]
