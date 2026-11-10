import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 112.0  # safety margin under 150s wall clock

    def time_ok():
        return time.time() < deadline

    # ---- primes up to a comfortable bound ----
    def sieve(n):
        is_prime = bytearray(b'\x01') * (n + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(n ** 0.5) + 1):
            if is_prime[i]:
                is_prime[i * i:: i] = bytearray(len(is_prime[i * i:: i]))
        return [i for i in range(n + 1) if is_prime[i]]

    primes = sieve(1000)

    # ---- fast exact scorer ----
    def divisors_from_factors(factors):
        divs = [1]
        for p, e in factors:
            cur = list(divs)
            mult = 1
            for _ in range(e):
                mult *= p
                cur = [d * mult for d in cur]
                divs.extend(cur)
        return divs

    def calc_score(factors):
        divs = divisors_from_factors(factors)
        if len(divs) < 2:
            return 0
        divs.sort()
        a = np.asarray(divs[:-1], dtype=np.int64)
        b = np.asarray(divs[1:], dtype=np.int64)
        g = np.gcd(a, b)
        return int(np.sum(g == 1))

    def product(factors):
        prod = 1
        for p, e in factors:
            prod *= p ** e
            if prod > 10 ** 18:
                return prod
        return prod

    def fits(factors):
        return product(factors) <= 10 ** 18

    # ---- search state ----
    best_score = 0
    best_factors = [(2, 1)]

    def consider(factors):
        nonlocal best_score, best_factors
        if not factors or not fits(factors):
            return
        s = calc_score(factors)
        if s > best_score:
            best_score = s
            best_factors = list(factors)

    # ---- 1. systematic base-2-exponent family ----
    # n = 2^e2 * prod(odd primes in a consecutive block)
    # Consecutive odd primes starting at index 1 (prime 3).
    for e2 in range(1, 10):
        if not time_ok():
            break
        # find how many odd primes fit
        # product of 2^e2 * 3*5*...
        # iterate count of odd primes
        prod = 2 ** e2
        maxcnt = 0
        for idx in range(1, len(primes)):
            prod *= primes[idx]
            if prod > 10 ** 18:
                break
            maxcnt = idx  # number of odd primes used = idx (indices 1..idx)
        # try counts from a few below maxcnt up to maxcnt
        for cnt in range(max(0, maxcnt - 6), maxcnt + 1):
            if not time_ok():
                break
            fac = [(2, e2)] + [(primes[i], 1) for i in range(1, 1 + cnt)]
            if fits(fac):
                consider(fac)

    # ---- 2. squarefree with first k primes ----
    for k in range(8, 17):
        if not time_ok():
            break
        fac = [(primes[i], 1) for i in range(k)]
        if fits(fac):
            consider(fac)

    # ---- 3. bump small-prime exponents on a good squarefree base ----
    for k in range(10, 16):
        if not time_ok():
            break
        base = [(primes[i], 1) for i in range(k)]
        if not fits(base):
            continue
        for e2 in range(2, 8):
            for e3 in range(1, 5):
                fac = list(base)
                fac[0] = (2, e2)
                if len(fac) > 1:
                    fac[1] = (3, e3)
                if fits(fac):
                    consider(fac)

    # ---- 4. local search from best ----
    rng = random.Random(12345)
    iterations = 0
    max_iters = 400000
    while time_ok() and iterations < max_iters:
        iterations += 1
        cur = list(best_factors)
        if not cur:
            break
        # pick a mutation
        mut = rng.random()
        trial = list(cur)

        if mut < 0.30:
            # bump a small-prime exponent
            idx = rng.randint(0, min(3, len(trial) - 1))
            p, e = trial[idx]
            ne = e + rng.choice([-1, 1, 1])
            if ne >= 1:
                trial[idx] = (p, ne)
        elif mut < 0.50:
            # add a new prime (small, not already present)
            present = set(p for p, _ in trial)
            maxp = max(p for p, _ in trial)
            cands = [p for p in primes if p > maxp and p not in present][:40]
            if cands:
                trial.append((rng.choice(cands), 1))
        elif mut < 0.65:
            # remove a larger prime
            if len(trial) > 6:
                idx = rng.randint(len(trial) // 2, len(trial) - 1)
                trial = trial[:idx] + trial[idx + 1:]
        elif mut < 0.85:
            # swap a prime for a nearby one (keep product similar)
            if len(trial) > 3:
                idx = rng.randint(1, len(trial) - 1)
                p, e = trial[idx]
                pos = primes.index(p)
                lo = max(1, pos - 3)
                hi = min(len(primes) - 1, pos + 3)
                opts = [primes[j] for j in range(lo, hi + 1) if primes[j] != p]
                if opts:
                    trial[idx] = (rng.choice(opts), e)
        else:
            # reduce a small-prime exponent to free budget, then maybe add prime
            idx = rng.randint(0, min(2, len(trial) - 1))
            p, e = trial[idx]
            if e > 1:
                trial[idx] = (p, e - 1)

        if trial == cur:
            continue
        if not fits(trial):
            # try to repair by reducing a small-prime exponent
            repaired = False
            for i in range(min(2, len(trial))):
                if trial[i][1] > 1:
                    p, e = trial[i]
                    trial[i] = (p, e - 1)
                    if fits(trial):
                        repaired = True
                        break
            if not repaired:
                continue

        s = calc_score(trial)
        if s > best_score:
            best_score = s
            best_factors = trial
        # occasional restart-like diversification: keep best, but allow occasional
        # acceptance of equal score to move around (helps escape plateaus)
        elif s == best_score and rng.random() < 0.02:
            best_factors = trial

    # ---- final construction ----
    n = 1
    for p, e in best_factors:
        n *= p ** e
    if n < 2:
        n = 2
    return n
