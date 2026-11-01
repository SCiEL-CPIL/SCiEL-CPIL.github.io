"""Genetic algorithm for the ternary covering code K_3(6,1), with deadline guard."""
import time
import random
import numpy as np

_DEADLINE = 150.0
_MARGIN = 10.0


def _to_val(word, q, n):
    v = 0
    for i in range(n):
        v = v * q + word[i]
    return v


def _to_word(val, q, n):
    w = [0] * n
    for i in range(n - 1, -1, -1):
        w[i] = val % q
        val //= q
    return w


def _precompute_diff(q, n):
    total = q ** n
    diff = np.empty((n, total, q), dtype=np.int8)
    for pos in range(n):
        for val in range(total):
            w = _to_word(val, q, n)
            for s in range(q):
                diff[pos, val, s] = 0 if w[pos] == s else 1
    return diff


def _coverage_mask(codes, q, n, radius, diff):
    total = q ** n
    arr = np.array([_to_val(c, q, n) for c in codes], dtype=np.int64)
    m = np.zeros(total, dtype=np.int32)
    for pos in range(n):
        for s in range(q):
            m += diff[pos, arr, s]
    return m <= radius


def _uncovered(codes, q, n, radius, diff):
    mask = _coverage_mask(codes, q, n, radius, diff)
    return int(np.count_nonzero(~mask)), mask


def _greedy_init(q, n, radius, diff, seed):
    rng = random.Random(seed)
    total = q ** n
    m = np.zeros(total, dtype=np.int32)
    uncovered = np.ones(total, dtype=bool)
    codes = []
    for _ in range(6):
        m = np.zeros(total, dtype=np.int32)
        for pos in range(n):
            for s in range(q):
                m += diff[pos, np.arange(total), s]
        gain = np.where(uncovered, m, 0)
        best = int(np.argmax(gain))
        if gain[best] <= 0:
            break
        codes.append(_to_word(best, q, n))
        arr = np.array([best], dtype=np.int64)
        for pos in range(n):
            for s in range(q):
                m += diff[pos, arr, s]
        uncovered = m > radius
    return codes


def _tetracode_init(q, n):
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    import itertools
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]


def construct(q, n, radius):
    start = time.time()
    deadline = start + _DEADLINE - _MARGIN
    total = q ** n
    diff = _precompute_diff(q, n)

    best_codes = _tetracode_init(q, n)
    best_unc, _ = _uncovered(best_codes, q, n, radius, diff)
    if best_unc == 0:
        return best_codes
    best_len = len(best_codes)

    rng = random.Random(12345)

    # build initial population
    pop = []
    pop.append(best_codes)
    for s in range(5):
        c = _greedy_init(q, n, radius, diff, 100 + s)
        pop.append(c)
    pop.append(best_codes[:])

    def fitness(codes):
        unc, _ = _uncovered(codes, q, n, radius, diff)
        return unc, len(codes)

    def crossover(a, b):
        # union of spheres, then trim random extras
        union = list(a) + list(b)
        # dedup
        seen = set()
        dedup = []
        for c in union:
            key = _to_val(c, q, n)
            if key not in seen:
                seen.add(key)
                dedup.append(c)
        # trim to a reasonable size
        target = min(len(a), len(b)) + rng.randint(0, 3)
        if len(dedup) > target:
            rng.shuffle(dedup)
            dedup = dedup[:target]
        return dedup

    def mutate(codes):
        c = list(codes)
        for _ in range(rng.randint(1, 4)):
            op = rng.random()
            if op < 0.4 and len(c) > 1:
                # remove a random codeword
                idx = rng.randrange(len(c))
                c.pop(idx)
            elif op < 0.7:
                # add a random codeword
                c.append([rng.randrange(q) for _ in range(n)])
            else:
                # replace a random codeword
                if c:
                    idx = rng.randrange(len(c))
                    c[idx] = [rng.randrange(q) for _ in range(n)]
        return c

    pop_size = 12
    while time.time() < deadline:
        # evaluate fitness
        scored = []
        for c in pop:
            scored.append(fitness(c))
        # sort by (unc, len)
        order = sorted(range(len(pop)), key=lambda i: scored[i])
        for i in order:
            if scored[i] < (best_unc, best_len):
                best_unc, best_len = scored[i]
                best_codes = list(pop[i])
                if best_unc == 0:
                    return best_codes

        # selection: keep top half, fill by offspring
        elite = [list(pop[i]) for i in order[:max(2, pop_size // 2)]]
        newpop = list(elite)
        while len(newpop) < pop_size and time.time() < deadline:
            i = order[rng.randrange(len(elite))]
            j = order[rng.randrange(len(elite))]
            child = crossover(pop[i], pop[j])
            child = mutate(child)
            if child:
                newpop.append(child)
        pop = newpop

    return best_codes
