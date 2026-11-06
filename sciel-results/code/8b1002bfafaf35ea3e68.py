import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    N = 1000000
    MAX_VAL = N + 1

    # Sieve of Eratosthenes to get primes up to sqrt(MAX_VAL)
    limit = int(math.isqrt(MAX_VAL))
    is_prime = [True] * (limit + 1)
    if limit >= 0:
        is_prime[0] = False
    if limit >= 1:
        is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False

    primes = [i for i, v in enumerate(is_prime) if v]

    # Initialize powerful part array using NumPy for vectorized updates
    P = np.ones(MAX_VAL + 1, dtype=np.int64)

    # Compute powerful parts using segmented sieve over primes
    # For each prime p, for each power p^k (k>=2), multiply P[m] by p^k
    # for all m that are divisible by p^k but not by p^(k+1)
    for p in primes:
        power = p * p
        while power <= MAX_VAL:
            next_power = power * p
            if next_power > MAX_VAL:
                # All multiples of 'power' in range have exactly this exponent
                # (since they can't be divisible by next_power)
                P[power::power] *= power
                break
            else:
                # Multiply by 'power' for multiples of 'power' that are NOT multiples of 'next_power'
                # Use boolean mask: divisible by power but not by next_power
                mask = np.zeros(MAX_VAL + 1, dtype=bool)
                mask[power::power] = True
                mask[next_power::next_power] = False
                P[mask] *= power
            power = next_power

            # Deadline check
            if time.time() > deadline:
                break

    # Check deadline before expensive scan
    if time.time() - start_time > deadline:
        return 10000

    # Find n in [10000, 1000000] maximizing P[n] * P[n+1]
    # Use NumPy for vectorized product scan
    products = P[10000:N+1] * P[10001:N+2]
    idx = int(np.argmax(products))
    best_n = 10000 + idx

    return best_n
