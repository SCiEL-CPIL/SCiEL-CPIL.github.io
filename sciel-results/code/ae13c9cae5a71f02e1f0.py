# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin before 90s kill

    N = 1000000
    MIN_N = 10000

    # Sieve of Eratosthenes to get primes up to sqrt(N+1)
    limit = int(math.isqrt(N + 1)) + 1
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            step = i
            start = i * i
            is_prime[start:limit + 1:step] = bytearray(len(range(start, limit + 1, step)))

    primes = [i for i in range(2, limit + 1) if is_prime[i]]

    # Array to store powerful parts
    pp = [1] * (N + 2)

    # For each prime, accumulate powerful part contributions
    # For prime p with p^2 <= N+1:
    #   For each power p^e (e >= 2) with p^e <= N+1:
    #     For each multiple m of p^e where m is NOT a multiple of p^(e+1):
    #       pp[m] *= p^e
    #
    # This is equivalent to: for each m divisible by p^2, multiply pp[m] by p^(v_p(m))
    #
    # Efficient implementation: iterate over powers of p, and for each power p^e,
    # multiply pp[m] by p for all multiples m of p^e (regardless of higher divisibility).
    # This gives pp[m] = product over p of p^(v_p(m)) for v_p(m) >= 2, which is correct.
    
    for p in primes:
        p2 = p * p
        if p2 > N + 1:
            continue

        # For each power p^e (e >= 2), multiply pp[m] by p for all m divisible by p^e
        # This accumulates p^(v_p(m)) for v_p(m) >= 2
        pe = p2
        while pe <= N + 1:
            # Multiply pp[m] by p for all m that are multiples of pe
            for m in range(pe, N + 2, pe):
                pp[m] *= p
            pe *= p
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    # Find n in [MIN_N, N] that maximizes pp[n] * pp[n+1]
    best_n = MIN_N
    best_val = pp[MIN_N] * pp[MIN_N + 1]

    for n in range(MIN_N, N):
        v = pp[n] * pp[n + 1]
        if v > best_val:
            best_val = v
            best_n = n
        if time.time() > deadline:
            break

    return best_n
# EVOLVE-BLOCK-END
