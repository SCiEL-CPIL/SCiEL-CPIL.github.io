import math
import time

def construct():
    start_time = time.time()
    deadline = 85  # safety margin before 90s kill
    N = 1000000
    MAX_VAL = N + 1  # need P[n+1] for n=N

    # Sieve of Eratosthenes for primes up to sqrt(MAX_VAL)
    limit = int(math.isqrt(MAX_VAL))
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = 0
    if limit >= 1:
        is_prime[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            step = i
            start = i * i
            is_prime[start:limit+1:step] = bytearray(len(range(start, limit+1, step)))
    primes = [i for i in range(2, limit + 1) if is_prime[i]]

    # Initialize powerful part array
    P = [1] * (MAX_VAL + 1)

    # Compute powerful parts via sieve
    for p in primes:
        power = p * p
        while power <= MAX_VAL:
            next_power = power * p
            if next_power > MAX_VAL:
                # All multiples of power get p^k (no higher power to exclude)
                for m in range(power, MAX_VAL + 1, power):
                    P[m] *= power
            else:
                # Only multiples of power NOT divisible by next_power have exact exponent k
                for m in range(power, MAX_VAL + 1, power):
                    if m % next_power != 0:
                        P[m] *= power
            power = next_power
            if time.time() - start_time > deadline - 8:
                break
        if time.time() - start_time > deadline - 8:
            break

    # Find n in [10000, 1000000] maximizing P[n] * P[n+1]
    best_n = 10000
    best_val = 0
    for n in range(10000, N + 1):
        val = P[n] * P[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
        if time.time() - start_time > deadline - 3:
            break

    return best_n
