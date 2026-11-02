import time
import itertools
import numpy as np

_DEADLINE = 150.0


def _base_codes(q, n, radius):
    """Initial seed set of codewords (distinct)."""
    if n == 4 and q == 3:
        return [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    if n >= 4 and q == 3:
        tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        return [h + list(tail) for h in tetracode
                for tail in itertools.product(range(q), repeat=n - 4)]
    # Fallback: all-zero plus a few simple words.
    z = [0] * n
    out = [list(z)]
    for i in range(n):
        for v in range(1, q):
            w = list(z)
            w[i] = v
            out.append(w)
    return out


def construct(q, n, radius):
    t0 = time.time()
    total = q ** n
    codes = _base_codes(q, n, radius)
    if len(codes) == 0:
        return [list(range(n)) % q]

    # Work with the full space if it is not too large; otherwise use a random sample
    # as a surrogate (the final code is still verified by the evaluator over the whole space).
    FULL = 300000
    if total <= FULL:
        idx = np.arange(total, dtype=np.int64)
        space = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
        space = space.reshape(total, n)
    else:
        m = min(200000, total)
        rng = np.random.RandomState(0)
        space = rng.randint(0, q, size=(m, n)).astype(np.int8)
        idx = rng.randint(0, total, size=m)

    # Vectorized "covered by one edit": for each position p, replace that position with
    # every other value -> total (q-1)*n neighbor indices.
    def neigh_codes(arr):
        npos = len(arr)
        if npos == 0:
            return np.zeros(0, dtype=np.int64)
        parts = []
        base = arr[:, 0].astype(np.int64) * (q ** (n - 1))
        for p in range(n):
            pw = q ** (n - 1 - p)
            other = np.arange(q, dtype=np.int64)
            for v in range(q):
                mask = arr[:, p] != v
                if not mask.any():
                    continue
                sub = arr[mask]
                if p == 0:
                    b = sub[:, 1].astype(np.int64)
                    for k in range(1, n - 1):
                        b = b * q + sub[:, 1 + k].astype(np.int64)
                    b = b * q + sub[:, n].astype(np.int64)
                    parts.append(base[mask] * (q - 1) + (q - 1 - v) * pw + b)
                else:
                    b = sub[:, 0].astype(np.int64)
                    for k in range(1, p):
                        b = b * q + sub[:, k].astype(np.int64)
                    for k in range(p + 1, n):
                        b = b * q + sub[:, k].astype(np.int64)
                    parts.append(base[mask] * (q - 1) + (q - 1 - v) * pw + b)
        if not parts:
            return np.zeros(0, dtype=np.int64)
        return np.concatenate(parts)

    # Coverage bitset over the sample.
    covered = np.zeros(len(space), dtype=bool)

    def apply(arr):
        if len(arr) == 0:
            return
        nb = neigh_codes(arr)
        hit = np.in1d(idx, nb)
        covered |= hit

    apply(np.array(codes, dtype=np.int8))
    cur = int(covered.sum())

    while cur < len(space):
        if time.time() > _DEADLINE - 20.0:
            break
        # Candidate pool: all words not yet in the code.
        pool = [list(w) for w in itertools.product(range(q), repeat=n)]
        pool = [w for w in pool if w not in codes]
        if not pool:
            break
        pool_arr = np.array(pool, dtype=np.int8)
        nb = neigh_codes(pool_arr)
        hit = np.in1d(idx, nb)
        gains = hit & (~covered)
        # Count gain per candidate.
        counts = np.bincount(nb[hit], minlength=len(space))
        # We need per-candidate gain: recompute via bincount of candidate index.
        cand_ids = np.repeat(np.arange(len(pool), dtype=np.int64),
                             [(q - 1) * n])
        cand_ids = cand_ids[hit]
        gains_per = np.bincount(cand_ids, minlength=len(pool))
        best = int(np.argmax(gains_per))
        if gains_per[best] <= 0:
            break
        codes.append(pool[best])
        apply(pool_arr[best:best + 1])
        cur = int(covered.sum())

    return codes
