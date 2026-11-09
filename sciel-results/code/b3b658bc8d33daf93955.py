import time
import math

def construct():
    start_time = time.time()
    DEADLINE = 85  # Safety margin before 90s limit

    MAX_N = 1000000
    MIN_N = 10000

    # Generate all powerful numbers up to MAX_N
    # A powerful number is of the form u^2 * v^3 where gcd(u,v) = 1
    # More simply: a number is powerful if in its prime factorization, every prime exponent >= 2
    # We can generate them by iterating over possible square parts and cube parts

    powerful = set()

    max_u = int(math.isqrt(MAX_N))
    max_v = int(MAX_N ** (1/3)) + 2

    for v in range(1, max_v + 1):
        v3 = v ** 3
        if v3 > MAX_N:
            break
        for u in range(1, max_u + 1):
            x = u * u * v3
            if x > MAX_N:
                break
            powerful.add(x)

    # Find the largest n in [MIN_N, MAX_N-1] such that both n and n+1 are powerful
    best_n = -1
    for n in range(MAX_N - 1, MIN_N - 1, -1):
        if n in powerful and (n + 1) in powerful:
            best_n = n
            break

    if best_n != -1:
        return best_n

    # Fallback: compute powerful parts for all numbers and maximize P(n)*P(n+1)
    # Powerful part of n = product of p^e for primes p with e >= 2 in factorization of n

    # Sieve for smallest prime factor
    spf = list(range(MAX_N + 1))
    for i in range(2, int(math.isqrt(MAX_N)) + 1):
        if spf[i] == i:
            for j in range(i * i, MAX_N + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # Compute powerful part for each number
    pow_part = [1] * (MAX_N + 1)
    for n in range(2, MAX_N + 1):
        x = n
        pp = 1
        while x > 1:
            p = spf[x]
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            if e >= 2:
                pp *= p ** e
        pow_part[n] = pp

    # Find n that maximizes pow_part[n] * pow_part[n+1]
    best_product = -1
    best_n_final = 10000

    # Check if we're still within time budget
    if time.time() - start_time > DEADLINE - 5:
        return 10000

    for n in range(MIN_N, MAX_N):
        product = pow_part[n] * pow_part[n + 1]
        if product > best_product:
            best_product = product
            best_n_final = n

    return best_n_final
