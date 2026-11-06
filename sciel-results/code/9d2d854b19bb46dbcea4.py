import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s margin, evaluator kills at 90s

    N_MAX = 1000000
    N_LIMIT = N_MAX + 1  # We need n+1 up to 1000001

    # 1. Sieve of Eratosthenes to find primes up to N_LIMIT
    is_prime = np.ones(N_LIMIT + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(N_LIMIT**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.nonzero(is_prime)[0].tolist()

    # 2. Precompute powerful parts using numpy
    val = np.ones(N_LIMIT + 1, dtype=np.int64)

    for p in primes:
        p2 = p * p
        if p2 > N_LIMIT:
            break

        # Iterate through powers p^k for k >= 2
        power = p2
        next_power = power * p

        while power <= N_LIMIT:
            # For each multiple m of 'power', check if exponent of p in m is exactly log_p(power)
            # i.e., m is divisible by power but not by next_power
            multiples = np.arange(power, N_LIMIT + 1, power)

            if next_power > N_LIMIT:
                # All multiples of power have exactly this exponent
                val[multiples] *= power
            else:
                # Filter: m % next_power != 0
                mask = (multiples % next_power) != 0
                filtered = multiples[mask]
                if len(filtered) > 0:
                    val[filtered] *= power

            power = next_power
            next_power *= p

            if time.time() > deadline:
                break

        if time.time() > deadline:
            break

    # 3. Find best n
    # Convert to Python list for the final search
    val_list = val.tolist()

    best_score = -1
    best_n = 10000

    for n in range(10000, N_MAX + 1):
        score = val_list[n] * val_list[n + 1]
        if score > best_score:
            best_score = score
            best_n = n

        if time.time() > deadline:
            break

    return best_n
