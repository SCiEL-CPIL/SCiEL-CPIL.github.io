import math
import numpy as np

def construct():
    N = 1000000
    START = 10000

    # 1. Sieve of Smallest Prime Factors (SPF) using NumPy for speed
    spf = np.arange(N + 1, dtype=np.int64)
    for i in range(2, int(math.isqrt(N)) + 1):
        if spf[i] == i:  # i is prime
            spf[i * i::i] = np.where(spf[i * i::i] == spf[i * i::i], i, spf[i * i::i])

    # 2. Compute Powerful Part for each number in range
    P = np.ones(N + 1, dtype=np.int64)

    for x in range(START, N + 1):
        temp = x
        p_part = 1
        while temp > 1:
            p = spf[temp]
            exp = 0
            while temp % p == 0:
                temp //= p
                exp += 1
            if exp >= 2:
                p_part *= (p ** exp)
        P[x] = p_part

    # 3. Find n that maximizes P[n] * P[n+1]
    # We can vectorize the multiplication for n in [START, N-1]
    products = P[START:N] * P[START+1:N+1]
    best_idx = np.argmax(products)
    best_n = START + best_idx

    return int(best_n)
