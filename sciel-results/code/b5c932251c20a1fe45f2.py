"""Search for n in [10000, 1000000] maximizing powerful_part(n)*powerful_part(n+1).

The powerful part of m is the product of full prime powers p^e with exponent e >= 2
in the prime factorization of m. We compute this for every m in the relevant range
using a smallest-prime-factor (SPF) table, then scan for the best consecutive pair.
"""
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85.0  # safety margin under the 90s kill
    N = 1000000
    LIMIT = N + 1  # need to factor up to N (n+1 for n = N-1 max is N)
    # We need powerful_part for n in [10000, N] and n+1 in [10001, N+1].
    # So factor up to N+1.
    max_val = N + 1

    # Build SPF table up to max_val
    spf = list(range(max_val + 1))
    spf[0] = 0
    spf[1] = 1
    for i in range(2, int(max_val ** 0.5) + 1):
        if spf[i] == i:  # i is prime
            step = i
            start = i * i
            for j in range(start, max_val + 1, step):
                if spf[j] == j:
                    spf[j] = i

    # Compute powerful part for each number 1..max_val
    # powerful_part[m] = product of p^e for prime powers with e >= 2
    pw = [1] * (max_val + 1)
    # For each number, factor using spf
    for m in range(2, max_val + 1):
        x = m
        prod = 1
        while x > 1:
            p = spf[x]
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            if e >= 2:
                prod *= p ** e
        pw[m] = prod

    # Check deadline; if exceeded, return a reasonable fallback
    if time.time() > deadline:
        return 10000

    # Scan for best n in [10000, N] maximizing pw[n] * pw[n+1]
    best_n = 10000
    best_val = pw[10000] * pw[10001]
    for n in range(10001, N):
        v = pw[n] * pw[n + 1]
        if v > best_val:
            best_val = v
            best_n = n

    return best_n
# EVOLVE-BLOCK-END
