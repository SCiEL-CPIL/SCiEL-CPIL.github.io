"""Evolved design: primes base + randomized multi-insertion search with restarts."""
import random
import time

# EVOLVE-BLOCK-START
def _block_products(seq):
    """Return set of products of all nonempty consecutive blocks of seq."""
    prods = set()
    n = len(seq)
    for i in range(n):
        p = 1
        for j in range(i, n):
            p *= seq[j]
            prods.add(p)
    return prods


def _try_insert(seq, x):
    """Try inserting x at a random position; return new seq if valid else None."""
    n = len(seq)
    pos = random.randint(0, n)
    cand = seq[:pos] + [x] + seq[pos:]
    return cand


def _valid(seq):
    """Check distinctness of all nonempty consecutive block products (incl. singletons)."""
    prods = set()
    n = len(seq)
    for i in range(n):
        p = 1
        for j in range(i, n):
            p *= seq[j]
            if p in prods:
                return False
            prods.add(p)
    return True


def _primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i in range(2, n + 1) if sieve[i]]


def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin before 90s wall-clock

    primes = _primes_upto(128)
    best = list(primes)
    best_len = len(best)

    rng = random.Random(123456789)

    # Precompute candidate composites (all non-primes in [2,128])
    prime_set = set(primes)
    composites = [x for x in range(2, 129) if x not in prime_set]

    def search(cur, cur_len, budget):
        """Repeatedly try random composites; return best (seq, len)."""
        local_best = list(cur)
        local_best_len = cur_len
        used = set(cur)
        while time.time() < budget:
            # pick a composite not already in the sequence
            pool = [c for c in composites if c not in used]
            if not pool:
                break
            x = rng.choice(pool)
            new_seq = _try_insert(local_best, x)
            if new_seq is not None and _valid(new_seq):
                local_best = new_seq
                local_best_len += 1
                used.add(x)
        return local_best, local_best_len

    # Phase 1: single global search with most of the budget
    phase1_budget = deadline - 20.0
    cand, cand_len = search(best, best_len, phase1_budget)
    if cand_len > best_len:
        best, best_len = cand, cand_len

    # Phase 2: restarts from primes with different seeds if time remains
    if time.time() < deadline - 5.0:
        for seed in range(2, 2000):
            if time.time() > deadline - 5.0:
                break
            rng2 = random.Random(seed * 7919 + 13)
            # temporarily rebind rng
            global_rng = rng
            rng = rng2
            b, bl = search(primes, len(primes), deadline - 5.0)
            if bl > best_len:
                best, best_len = b, bl
            rng = global_rng

    return best
# EVOLVE-BLOCK-END
