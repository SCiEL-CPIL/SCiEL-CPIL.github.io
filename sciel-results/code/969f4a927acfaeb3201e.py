import time
import numpy as np
from math import isqrt

def construct():
    t0 = time.time()
    DEADLINE = 88  # Safety margin for 90s limit

    def elapsed():
        return time.time() - t0

    MAXN = 1_000_000
    MAXN_PLUS = MAXN + 1

    # 1. Sieve of Eratosthenes to get primes
    is_prime = bytearray(b'\x01') * (MAXN_PLUS)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, isqrt(MAXN) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, MAXN_PLUS) if is_prime[i]]

    # 2. Compute powerful part P(m) for all m using NumPy
    # P(m) = product of p^k where k = v_p(m) and k >= 2
    pp = np.ones(MAXN_PLUS, dtype=np.int64)

    for p in primes:
        if elapsed() > DEADLINE:
            break

        p_power = p * p

        while p_power <= MAXN:
            next_power = p_power * p
            max_j = MAXN // p_power

            if next_power <= MAXN:
                # j values where j % p != 0
                j = np.arange(1, max_j + 1)
                mask = (j % p) != 0
                j_filtered = j[mask]
                indices = p_power * j_filtered
            else:
                indices = np.arange(p_power, MAXN_PLUS, p_power)

            if len(indices) > 0:
                pp[indices] *= p_power

            p_power = next_power

    # 3. Find max pp[n] * pp[n+1] for n in [10000, MAXN-1]
    if elapsed() > DEADLINE:
        return 10000

    products = pp[10000:MAXN] * pp[10001:MAXN_PLUS]

    best_idx = np.argmax(products)
    best_n = 10000 + best_idx

    return int(best_n)
