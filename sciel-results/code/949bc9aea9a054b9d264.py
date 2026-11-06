import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 90s kill
    TARGET = 123457

    # Compute largest prime factor for all n in [1, TARGET] using a sieve.
    # lpf[n] = largest prime factor of n. lpf[1] = 1.
    lpf = [0] * (TARGET + 1)
    lpf[1] = 1

    # For each prime i, mark all multiples. Since we iterate i in increasing order,
    # the last prime that divides j will be the largest.
    for i in range(2, TARGET + 1):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, TARGET + 1, i):
                lpf[j] = i

    # Now scan all decompositions a + b = TARGET, a >= 1, b >= 1
    best_max_lpf = TARGET + 1
    best_a = 1
    best_b = TARGET - 1

    for a in range(1, TARGET):
        b = TARGET - a
        m = lpf[a] if lpf[a] > lpf[b] else lpf[b]
        if m < best_max_lpf:
            best_max_lpf = m
            best_a = a
            best_b = b

    # Check deadline - if we've found a good answer, return it
    # The linear scan is O(TARGET) which is ~123k iterations, very fast
    return [best_a, best_b]
