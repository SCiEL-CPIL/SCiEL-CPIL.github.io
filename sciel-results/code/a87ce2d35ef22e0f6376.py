"""MODEL-AND-OPTIMIZE: greedy weighted set-packing over prime-factor sets for pairwise-coprime max-sum."""
import math
import time

# EVOLVE-BLOCK-START
N = 256

def _primes_up_to(n):
    if n < 2:
        return []
    sieve = bytearray([1]) * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i:n + 1:i] = bytearray(len(range(i * i, n + 1, i)))
    return [i for i in range(2, n + 1) if sieve[i]]

def _prime_factors(x):
    """Return the set of distinct prime factors of x."""
    fs = set()
    d = 2
    while d * d <= x:
        if x % d == 0:
            fs.add(d)
            while x % d == 0:
                x //= d
        d += 1
    if x > 1:
        fs.add(x)
    return fs

def construct():
    start = time.time()
    deadline = start + 55.0  # safety margin under 60s

    primes = _primes_up_to(N)
    prime_set = set(primes)

    # Precompute prime-factor set for each number 1..256
    pf = {}
    for x in range(1, N + 1):
        pf[x] = _prime_factors(x)

    # Candidate numbers sorted descending by value
    candidates = list(range(N, 0, -1))  # 256, 255, ..., 1

    # Greedy weighted set-packing: pick the largest-value number whose
    # prime factors are all unused. Include 1 (empty factor set) last.
    used_primes = set()
    result = []

    for x in candidates:
        if time.time() > deadline:
            break
        factors = pf[x]
        if not factors.issubset(prime_set):
            # x has a prime factor > 256 (impossible for x <= 256, but safe)
            continue
        if factors & used_primes:
            # conflicts with an already-selected number
            continue
        # x is coprime to every selected number
        result.append(x)
        used_primes |= factors

    # 1 has no prime factors, so it's always compatible; ensure it's included
    if 1 not in result:
        result.append(1)

    # Verification: pairwise coprime, distinct, in range, Python int
    n_res = len(result)
    valid = True
    for i in range(n_res):
        for j in range(i + 1, n_res):
            if math.gcd(result[i], result[j]) != 1:
                valid = False
                break
        if not valid:
            break

    if not (valid and len(set(result)) == len(result) and all(1 <= x <= N for x in result)):
        # Fallback: 1 plus all primes
        result = [1] + primes[:]

    # Type-check: ensure Python ints, not bools or numpy types
    final = [int(x) for x in result if isinstance(x, int) and not isinstance(x, bool)]
    if not final:
        final = [1]

    return final
# EVOLVE-BLOCK-END
