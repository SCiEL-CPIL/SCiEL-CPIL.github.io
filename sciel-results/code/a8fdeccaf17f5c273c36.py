"""Greedy randomized set-cover for the ternary covering code K_3(7,1)."""
import itertools
import time
import random


def construct(q, n, radius):
    start = time.time()
    deadline = start + 140.0  # safety margin under the 150s kill

    if (q, n, radius) != (3, 7, 1):
        return _fallback(q, n, radius)

    qpow = [1] * (n + 1)
    for i in range(1, n + 1):
        qpow[i] = qpow[i - 1] * q
    total = qpow[n]

    def idx_to_word(idx):
        w = [0] * n
        for pos in range(n - 1, -1, -1):
            w[pos] = idx % q
            idx //= q
        return w

    # Covered set for each candidate index (radius-1 ball).
    covered_sets = []
    for idx in range(total):
        w = idx_to_word(idx)
        s = {idx}
        for pos in range(n):
            old = w[pos]
            for v in range(q):
                if v == old:
                    continue
                w[pos] = v
                ii = 0
                for i in range(n):
                    ii += w[i] * qpow[i]
                s.add(ii)
            w[pos] = old
        covered_sets.append(sorted(s))

    # Global uncovered bitset (list of 64-bit words).
    nwords = (total + 63) // 64
    uncovered = [(1 << 64) - 1] * nwords
    if total % 64 != 0:
        rem = total % 64
        uncovered[-1] = (1 << rem) - 1

    # Per-candidate bit list for fast new-count and apply.
    cover_bits = []
    for idx in range(total):
        lst = []
        for j in covered_sets[idx]:
            lst.append((j >> 6, 1 << (j & 63)))
        cover_bits.append(lst)

    def count_new(idx):
        c = 0
        for (widx, bit) in cover_bits[idx]:
            if uncovered[widx] & bit:
                c += 1
        return c

    def apply(idx):
        for (widx, bit) in cover_bits[idx]:
            uncovered[widx] &= ~bit

    def greedy_cover(seed):
        rnd = random.Random(seed)
        chosen = []
        total_uncovered = total
        pool = list(range(total))
        while total_uncovered > 0:
            best_c = -1
            ties = []
            for idx in pool:
                c = count_new(idx)
                if c > best_c:
                    best_c = c
                    ties = [idx]
                elif c == best_c:
                    ties.append(idx)
            if best_c <= 0:
                break
            best = rnd.choice(ties)
            chosen.append(best)
            apply(best)
            total_uncovered -= best_c
            pool.remove(best)
            if time.time() > deadline:
                break
        return chosen

    def is_valid(chosen):
        if not chosen:
            return False
        if len(set(chosen)) != len(chosen):
            return False
        cov = bytearray(total)
        for idx in chosen:
            for j in covered_sets[idx]:
                cov[j] = 1
        for j in range(total):
            if not cov[j]:
                return False
        return True

    best_code = None
    code = greedy_cover(0)
    if is_valid(code):
        best_code = code
    seed = 1
    while time.time() < deadline - 5.0:
        code = greedy_cover(seed)
        if is_valid(code):
            if best_code is None or len(code) < len(best_code):
                best_code = code
        seed += 1

    if best_code is None:
        best_code = _direct_sum(q, n, radius)

    return [idx_to_word(idx) for idx in best_code]


def _direct_sum(q, n, radius):
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]


def _fallback(q, n, radius):
    if q == 3 and n >= 4:
        return _direct_sum(q, n, radius)
    return [list(w) for w in itertools.product(range(q), repeat=n)]
