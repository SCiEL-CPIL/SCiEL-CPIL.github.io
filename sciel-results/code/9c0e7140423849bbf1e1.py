import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    N = 1000000
    MAX_VAL = N + 1

    limit = int(math.isqrt(MAX_VAL))
    is_prime = bytearray(b'\x01') * (limit + 1)
    if limit >= 0:
        is_prime[0] = 0
    if limit >= 1:
        is_prime[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = 0

    primes = [i for i in range(2, limit + 1) if is_prime[i]]

    P = np.ones(MAX_VAL + 1, dtype=np.int64)

    for p in primes:
        power = p * p
        while power <= MAX_VAL:
            if time.time() > deadline:
                break
            next_power = power * p
            if next_power > MAX_VAL:
                P[power::power] *= power
                break
            else:
                max_j = MAX_VAL // power
                j_vals = np.arange(1, max_j + 1)
                mask = j_vals % p != 0
                indices = power * j_vals[mask]
                P[indices] *= power
            power = next_power

    if time.time() - start_time > deadline:
        return 10000

    products = P[10000:N + 1] * P[10001:N + 2]
    idx = int(np.argmax(products))
    best_n = 10000 + idx

    return best_n
