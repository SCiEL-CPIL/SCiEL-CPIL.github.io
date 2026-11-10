import time
import math

def construct():
    start = time.time()
    DEADLINE = 60.0  # safety margin before the 90s kill

    N = 1000000
    NMAX = N + 1  # need lpf for values up to 1000001

    # Sieve of Eratosthenes to find primes up to NMAX
    is_prime = bytearray(b'\x01') * (NMAX + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, math.isqrt(NMAX) + 1):
        if is_prime[i]:
            is_prime[i * i: NMAX + 1: i] = bytearray(len(is_prime[i * i: NMAX + 1: i]))

    # Collect primes
    primes = [i for i in range(2, NMAX + 1) if is_prime[i]]

    # Compute largest prime factor for each number up to NMAX
    lpf = [0] * (NMAX + 1)
    for p in primes:
        for m in range(p, NMAX + 1, p):
            lpf[m] = p

    # Find n in [10000, 1000000] minimizing max(lpf[n], lpf[n+1])
    best_n = 10000
    best_val = max(lpf[10000], lpf[10001])

    for n in range(10001, N + 1):
        val = max(lpf[n], lpf[n + 1])
        if val < best_val:
            best_val = val
            best_n = n

        # Time check
        if time.time() - start > DEADLINE - 5.0:
            break

    return best_n
