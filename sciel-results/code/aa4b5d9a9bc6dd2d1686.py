import time

N = 5000
DEADLINE = 300.0  # safety margin vs the 330s kill


def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]


def _is_admissible(res, primes):
    L = len(res)
    if L == 0:
        return True
    present = {}
    for p in primes:
        if p > L:
            break
        seen = set()
        for x in res:
            seen.add(x % p)
        if len(seen) == p:
            return False
    return True


def construct():
    start = time.time()

    def now():
        return time.time()

    primes = _primes(N)

    # Per-prime residue counts and number of *present* (nonzero) residues.
    counts = {p: [0] * p for p in primes}
    present = {p: 0 for p in primes}
    in_set = [True] * (N + 1)

    # Initialise from the full set [0, N].
    for x in range(N + 1):
        for p in primes:
            r = x % p
            if counts[p][r] == 0:
                present[p] += 1
            counts[p][r] += 1

    L = N + 1

    def remove(x):
        nonlocal L
        in_set[x] = False
        L -= 1
        for p in primes:
            r = x % p
            counts[p][r] -= 1
            if counts[p][r] == 0:
                present[p] -= 1

    def add(x):
        nonlocal L
        in_set[x] = True
        L += 1
        for p in primes:
            r = x % p
            if counts[p][r] == 0:
                present[p] += 1
            counts[p][r] += 1

    # ---- Phase 1: greedy removal until admissible ----
    # For the smallest violating prime, drop the minimum-count residue class.
    while now() - start < DEADLINE - 40:
        viol = None
        for p in primes:
            if p > L:
                break
            if present[p] == p:  # all residues present -> must break this prime
                viol = p
                break
        if viol is None:
            break
        c = counts[viol]
        m = min(c)
        r = c.index(m)
        for x in range(N + 1):
            if in_set[x] and x % viol == r:
                remove(x)

    # ---- Phase 2: add-back local improvement ----
    # Adding x is safe iff no prime p <= L+1 has exactly one missing residue
    # equal to x % p.
    def safe(x):
        limit = L + 1
        for p in primes:
            if p > limit:
                break
            if present[p] == p - 1 and counts[p][x % p] == 0:
                return False
        return True

    # Multiple passes: each pass may unlock further additions.
    for _ in range(500):
        if now() - start > DEADLINE - 25:
            break
        added = False
        for x in range(N + 1):
            if in_set[x]:
                continue
            if safe(x):
                add(x)
                added = True
        if not added:
            break

    res = [x for x in range(N + 1) if in_set[x]]

    # ---- Final safety: ensure admissibility before returning ----
    if not _is_admissible(res, primes):
        # Rebuild tracking state from res and finish a greedy removal.
        for p in primes:
            counts[p] = [0] * p
            present[p] = 0
        in_set = [False] * (N + 1)
        for x in res:
            in_set[x] = True
            for p in primes:
                r = x % p
                if counts[p][r] == 0:
                    present[p] += 1
                counts[p][r] += 1
        L = len(res)
        while now() - start < DEADLINE - 5:
            viol = None
            for p in primes:
                if p > L:
                    break
                if present[p] == p:
                    viol = p
                    break
            if viol is None:
                break
            c = counts[viol]
            m = min(c)
            r = c.index(m)
            for x in range(N + 1):
                if in_set[x] and x % viol == r:
                    remove(x)
        res = [x for x in range(N + 1) if in_set[x]]

    return res
