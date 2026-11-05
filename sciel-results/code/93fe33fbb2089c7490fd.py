"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def construct():
    N = 1000000
    MIN_N = 10000
    deadline = time.time() + 80  # 80s safety margin

    # 1. Sieve for Smallest Prime Factor (SPF)
    spf = list(range(N + 1))
    for i in range(2, int(math.isqrt(N)) + 1):
        if spf[i] == i:
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # 2. Precompute powerful part for all numbers up to N
    pp = [1] * (N + 1)
    for n in range(2, N + 1):
        if time.time() > deadline:
            break
        m = n
        prod = 1
        while m > 1:
            p = spf[m]
            cnt = 0
            while m % p == 0:
                m //= p
                cnt += 1
            if cnt >= 2:
                prod *= (p ** cnt)
        pp[n] = prod

    # 3. Find max product of consecutive powerful parts
    best_n = MIN_N
    best_val = 0
    for n in range(MIN_N, N):
        if time.time() > deadline:
            break
        val = pp[n] * pp[n + 1]
        if val > best_val:
            best_val = val
            best_n = n

    return best_n

# EVOLVE-BLOCK-END
