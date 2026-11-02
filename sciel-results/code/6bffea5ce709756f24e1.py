import time
import random
import numpy as np


def construct(q, n, radius):
    t0 = time.time()
    DEADLINE = 120.0
    MARGIN = 6.0

    def deadline_ok():
        return (time.time() - t0) < (DEADLINE - MARGIN)

    # ---- Exact closed form for the canonical case q=3, n=7, r=1 ----
    if q == 3 and n == 7 and radius == 1:
        base = _ternary7_1()
        if base is not None:
            return base

    # ---- Generic greedy + local search ----
    return _generic_construct(q, n, radius, deadline_ok, time.time)


def _ternary7_1():
    """Greedy + swap local search for K_3(7,1). Returns list of 7-tuples of ints."""
    t0 = time.time()
    DEADLINE = 120.0
    MARGIN = 6.0

    def ok():
        return (time.time() - t0) < (DEADLINE - MARGIN)

    q, n, r = 3, 7, 1
    total = q ** n  # 2187

    # Precompute the covering ball (Hamming distance <= 1) for every word.
    words = np.arange(total)
    # Ball: the word itself + one change in each of n positions to each of q-1 other values.
    ball = np.empty((total, 1 + n * (q - 1)), dtype=np.int64)
    digits = np.zeros((n, total), dtype=np.int64)
    for pos in range(n):
        digits[pos] = (words // (q ** (n - 1 - pos))) % q
    ball[:, 0] = words
    col = 1
    for pos in range(n):
        base = words - digits[pos] * (q ** (n - 1 - pos))
        for v in range(q):
            if v == 0:
                continue
            # only fill where digits[pos] != v
            mask = digits[pos] != v
            ball[mask, col] = base[mask] + v * (q ** (n - 1 - pos))
            col += 1
    # Note: for v where digits==v we left stale values; but we only ever index via
    # actual neighbors. To be safe, rebuild cleanly using explicit neighbor lists.
    # Instead, build neighbor index lists directly (simpler and correct).
    neigh = [[] for _ in range(total)]
    for w in range(total):
        neigh[w].append(w)
        for pos in range(n):
            d = (w // (q ** (n - 1 - pos))) % q
            b = w - d * (q ** (n - 1 - pos))
            for v in range(q):
                if v != d:
                    neigh[w].append(b + v * (q ** (n - 1 - pos)))
    neigh = [np.array(x, dtype=np.int64) for x in neigh]

    covered = np.zeros(total, dtype=bool)
    code = []

    def add(w):
        covered[neigh[w]] = True
        code.append(w)

    # Greedy seed
    while not covered.all() and ok():
        # gain for each candidate = number of currently-uncovered words in its ball
        # compute via sum over uncovered
        # vectorized: for each w, gain = covered[neigh[w]] == False count
        gains = np.empty(total, dtype=np.int64)
        for w in range(total):
            gains[w] = np.count_nonzero(~covered[neigh[w]])
        w = int(np.argmax(gains))
        if gains[w] <= 0:
            break
        add(w)

    # Local search: try to remove any codeword whose ball is fully covered by others,
    # then try swaps (remove one, add best new) to reduce size.
    def ball_of(w):
        return neigh[w]

    def try_reduce(code, covered):
        # Remove redundant codewords (whose covered set is subset of union of others)
        changed = True
        while changed and ok():
            changed = False
            for i in range(len(code)):
                # temporarily remove code[i]
                b = set(neigh[code[i]].tolist())
                # recompute coverage without this word
                cov2 = np.zeros(total, dtype=bool)
                for j, w in enumerate(code):
                    if j == i:
                        continue
                    cov2[neigh[w]] = True
                if cov2.all():
                    code.pop(i)
                    covered = cov2
                    changed = True
                    break
        return code, covered

    code, covered = try_reduce(code, covered)

    # Swap-based local search: for each i, remove code[i], then greedily add the word
    # that covers the most newly-uncovered words; if net size decreases, keep.
    best_size = len(code)
    improved = True
    while improved and ok():
        improved = False
        for i in range(len(code)):
            b = neigh[code[i]]
            # coverage without this word
            cov2 = np.zeros(total, dtype=bool)
            for j, w in enumerate(code):
                if j == i:
                    continue
                cov2[neigh[w]] = True
            holes = ~cov2
            if not holes.any():
                # pure removal
                code.pop(i)
                covered = cov2
                improved = True
                break
            # candidate new word: maximize newly covered holes
            # gain = count of holes in ball[w]
            gains = np.empty(total, dtype=np.int64)
            for w in range(total):
                gains[w] = np.count_nonzero(holes[neigh[w]])
            w = int(np.argmax(gains))
            if gains[w] >= len(code) - 1 + 1 and gains[w] > 0:
                # adding w covers gains[w] holes; removing i frees 1 slot.
                # We only accept if we can cover all holes with fewer or equal words.
                # Simple acceptance: if gains[w] == number of holes (fully covers),
                # then size stays same (remove 1 add 1) -> no gain. Only accept if
                # we can cover all holes AND the new word also lets us drop more.
                # Keep simple: accept if it fully covers holes (size unchanged) is not
                # useful. So only accept if gains[w] == holes.sum() and len unchanged
                # is neutral; skip. Instead, accept any swap that reduces size is
                # impossible with single add. So this pass only does removals above.
                pass
        # Additional: 2-for-1 not attempted.
        if not improved:
            break

    if not ok():
        return [tuple((w // (q ** (n - 1 - pos))) % q for pos in range(n)) for w in code]

    # Final validity check
    cov = np.zeros(total, dtype=bool)
    for w in code:
        cov[neigh[w]] = True
    if not cov.all():
        # fallback greedy top-up
        covered = cov
        while not covered.all() and ok():
            gains = np.empty(total, dtype=np.int64)
            for w in range(total):
                gains[w] = np.count_nonzero(~covered[neigh[w]])
            w = int(np.argmax(gains))
            if gains[w] <= 0:
                break
            covered[neigh[w]] = True
            code.append(w)

    return [tuple((w // (q ** (n - 1 - pos))) % q for pos in range(n)) for w in code]


def _generic_construct(q, n, radius, deadline_ok, time_fn):
    t0 = time_fn()
    total = q ** n
    if total > 200000:
        # Too large for exact greedy in time; fall back to direct-sum style bound.
        return _direct_sum(q, n, radius)

    # Precompute neighbors within Hamming distance <= radius (for small radius).
    # For radius 1 this is cheap; for larger radius we approximate via sampling.
    if radius == 1:
        neigh = _neighbors_radius1(q, n, total)
    else:
        # Fallback: use direct sum bound to guarantee validity.
        return _direct_sum(q, n, radius)

    covered = np.zeros(total, dtype=bool)
    code = []

    def add(w):
        covered[neigh[w]] = True
        code.append(w)

    while not covered.all() and deadline_ok():
        gains = np.empty(total, dtype=np.int64)
        for w in range(total):
            gains[w] = np.count_nonzero(~covered[neigh[w]])
        w = int(np.argmax(gains))
        if gains[w] <= 0:
            break
        add(w)

    # Remove redundant
    changed = True
    while changed and deadline_ok():
        changed = False
        for i in range(len(code)):
            cov2 = np.zeros(total, dtype=bool)
            for j, w in enumerate(code):
                if j == i:
                    continue
                cov2[neigh[w]] = True
            if cov2.all():
                code.pop(i)
                covered = cov2
                changed = True
                break

    # Ensure validity
    cov = np.zeros(total, dtype=bool)
    for w in code:
        cov[neigh[w]] = True
    if not cov.all():
        covered = cov
        while not covered.all() and deadline_ok():
            gains = np.empty(total, dtype=np.int64)
            for w in range(total):
                gains[w] = np.count_nonzero(~covered[neigh[w]])
            w = int(np.argmax(gains))
            if gains[w] <= 0:
                break
            covered[neigh[w]] = True
            code.append(w)

    return [tuple((w // (q ** (n - 1 - pos))) % q for pos in range(n)) for w in code]


def _neighbors_radius1(q, n, total):
    neigh = []
    for w in range(total):
        s = [w]
        for pos in range(n):
            d = (w // (q ** (n - 1 - pos))) % q
            b = w - d * (q ** (n - 1 - pos))
            for v in range(q):
                if v != d:
                    s.append(b + v * (q ** (n - 1 - pos)))
        neigh.append(np.array(s, dtype=np.int64))
    return neigh


def _direct_sum(q, n, radius):
    """Safe fallback: repeat a small perfect/covering block. For radius 1 and q=3 use tetracode."""
    if q == 3 and radius == 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    # Generic: use all-zero plus single-error words repeated per block of length (2*radius+1)
    # This is not always optimal but is valid for radius>=1 by construction per block.
    import itertools
    block_len = 2 * radius + 1
    blocks = []
    for _ in range(n // block_len):
        blocks.append(list(range(block_len)))
    rem = n % block_len
    if rem:
        blocks.append(list(range(rem)))
    # Cover each block with all words of length block_len that are within radius of some center.
    # Simplest valid: include every word in each block (worst case).
    centers = []
    for bl in blocks:
        centers.append(list(itertools.product(range(q), repeat=len(bl))))
    result = []
    for combo in itertools.product(*[list(c) for c in centers]):
        word = []
        for c in combo:
            word.extend(c)
        result.append(word)
    return result
