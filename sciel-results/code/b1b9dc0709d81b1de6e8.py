"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    N = 1000001  # we need indices up to 1000000, and n+1 up to 1000001

    # Multiplicative sieve for powerful part P(n)
    # P(n) = product of p^e for each prime p where e >= 2 in factorization of n
    # For each prime p, for each power p^k with k>=2, multiply p^k into P(m)
    # for every multiple m of p^k.
    #
    # But careful: if n = p^4 * q, then P(n) should include p^4 (the FULL power).
    # So for prime p, the contribution to P(m) is p^e where e is the full exponent
    # of p in m, if e >= 2.
    #
    # Sieve approach:
    # 1. Initialize P[0..N] = 1
    # 2. For each prime p:
    #    For each power pk = p^k (k>=2) up to N:
    #      For each multiple m of pk:
    #        We need to determine the exact exponent of p in m.
    #
    # Better approach:
    # For each prime p, find all m that have p^e | m with e>=2.
    # The contribution is p^e where e is the exact exponent.
    #
    # Efficient sieve:
    # For each prime p:
    #   p2 = p*p
    #   For each multiple m of p2:
    #     Determine the exact exponent e of p in m.
    #     Multiply P[m] by p^e.
    #
    # To find exact exponent: for m divisible by p^2, count how many times p divides m.
    # But doing this naively is slow. Instead:
    #
    # Alternative: For each prime p, iterate through powers p^k (k=2,3,...)
    # and for multiples of p^k but NOT p^(k+1), multiply by p^k.
    #
    # Even better: For each prime p, for each multiple m of p^2:
    #   e = 0
    #   temp = m
    #   while temp % p == 0:
    #       temp //= p
    #       e += 1
    #   if e >= 2:
    #       P[m] *= p^e
    #
    # This is still O(N * sum of exponents) which could be slow.
    #
    # Fastest approach: For each prime p, compute p^2, p^3, ... up to N.
    # For each power pk = p^k (k >= 2), for each multiple m of pk:
    #   if m is NOT divisible by p^(k+1), then the exponent of p in m is exactly k,
    #   so multiply P[m] by p^k.
    #   But wait, m could have higher exponent. We need the EXACT exponent.
    #
    # Let me think again. The cleanest sieve:
    # For each prime p:
    #   For m from p^2 to N, step p^2:
    #     Count the exponent of p in m by dividing.
    #
    # Actually, let's use a different decomposition:
    # P(m) = product over primes p | m of (p^e_p(m) if e_p(m) >= 2 else 1)
    #
    # Sieve:
    # P = [1] * (N+1)
    # For each prime p:
    #   p2 = p * p
    #   if p2 > N: break
    #   For m in range(p2, N+1, p2):
    #     # m is divisible by p^2
    #     # Find exact exponent of p in m
    #     e = 2
    #     temp = m
    #     while temp % p == 0:
    #         temp //= p
    #         e += 1
    #     # e is now the exact exponent
    #     P[m] *= p ** e
    #
    # But the inner while loop makes this slow. Let's optimize:
    # For each prime p, precompute the powers p^2, p^3, ..., p^max.
    # For m divisible by p^k but not p^(k+1), the contribution is p^k.
    #
    # So: For each prime p:
    #   powers = [p^2, p^3, ..., p^max] where p^max <= N
    #   For k from max down to 2:
    #     pk = p^k
    #     For m in range(pk, N+1, pk):
    #       if m % (p * pk) != 0:  # m is not divisible by p^(k+1)
    #         P[m] *= pk
    #
    # This works because if m is divisible by p^k but not p^(k+1), then e_p(m) = k.
    # We process from high k to low k so each m gets its contribution exactly once.
    #
    # Wait, no. If m is divisible by p^5, then when k=5, m % p^6 != 0 (assuming p^6 > m or m not div by p^6), so we add p^5. Good.
    # If m is divisible by p^3 but not p^4, when k=3, m % p^4 != 0, so we add p^3. When k=4, m is not in range(p^4, ...) so we skip. Good.
    #
    # This is correct! And it avoids the inner while loop.

    P = [1] * (N + 1)

    # Generate primes up to N using simple sieve
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i * i, N + 1, i):
                is_prime[j] = False

    primes = [i for i in range(2, N + 1) if is_prime[i]]

    for p in primes:
        p2 = p * p
        if p2 > N:
            break
        # Find all powers p^k for k >= 2
        pk = p2
        powers = []
        while pk <= N:
            powers.append(pk)
            pk *= p
        # Process from highest power to lowest
        for k in range(len(powers) - 1, -1, -1):
            pk = powers[k]
            p_next = pk * p  # p^(k+2)
            for m in range(pk, N + 1, pk):
                if m % p_next != 0:
                    P[m] *= pk

    # Now P[n] is the powerful part of n for all n in [0, N]
    # Find n in [10000, 1000000] maximizing P[n] * P[n+1]
    best_n = 10000
    best_score = 0

    for n in range(10000, 1000001):
        score = P[n] * P[n + 1]
        if score > best_score:
            best_score = score
            best_n = n

        if time.time() - start_time > deadline:
            break

    return best_n
# EVOLVE-BLOCK-END
