"""Greedy covering-code search for K_q(n, radius), with a direct-sum fallback."""
import time
import itertools
import numpy as np

_DEADLINE_FRACTION = 0.90  # stop well before the 150s kill


def _tetracode():
    return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]


def _direct_sum(q, n, radius):
    # Direct sum of the ternary tetracode (9 words, length 4, radius-1 perfect) with
    # every word on the remaining n - 4 coordinates.
    if n < 4:
        return [list(t) for t in itertools.product(range(q), repeat=n)]
    tetracode = _tetracode()
    return [h + list(tail) for h in tetracode
            for tail in itertools.product(range(q), repeat=n - 4)]


def _is_covering(codewords, q, n, radius):
    # Exact check: every word is within Hamming distance <= radius of some codeword.
    m = len(codewords)
    if m == 0:
        return n == 0
    cw = np.array(codewords, dtype=np.uint8)
    for word in itertools.product(range(q), repeat=n):
        w = np.fromiter(word, dtype=np.uint8, count=n)
        diffs = (cw != w[None, :])
        d = diffs.sum(axis=1)
        if d.min() > radius:
            return False
    return True


def _ball_masks(q, n, radius):
    # masks[i] = bitset of words at Hamming distance <= radius from word i.
    N = q ** n
    Nw = (N + 7) // 8
    masks = np.empty((N, Nw), dtype=np.uint64)
    words = np.frombuffer(np.arange(N, dtype=np.uint64).tobytes(), dtype=np.uint8)
    words = words.reshape(N, 1)
    for r in range(radius + 1):
        for pos in itertools.combinations(range(n), r):
            pos_arr = np.array(pos, dtype=np.intp)
            # new value = (word + 1 + k) mod q for each k = 1..q-1
            for k in range(1, q):
                shifted = (words + k) % q
                shifted[:, pos_arr] = (words[:, pos_arr] + k) % q
                idx = shifted.ravel()
                masks |= np.take(masks, idx, axis=0)
    return masks


def _random_greedy(q, n, radius, masks, order, deadline, rng):
    N = q ** n
    Nw = (N + 7) // 8
    covered = np.zeros(Nw, dtype=np.uint64)
    remaining = N
    picked = []
    while remaining > 0:
        if time.time() >= deadline:
            return None
        # score = count of still-uncovered words in the ball
        best_i = -1
        best_score = 0
        for i in order:
            m = masks[i]
            s = np.bitwise_and(m, ~covered).count_nonzero()
            if s > best_score:
                best_score = s
                best_i = i
                if best_score == remaining:  # perfect: covers everything
                    break
        if best_i < 0 or best_score <= 0:
            return None
        picked.append(best_i)
        covered |= masks[best_i]
        remaining = N - covered.count_nonzero()
    return picked


def _greedy_search(q, n, radius, deadline, rng):
    N = q ** n
    if N <= 0:
        return []
    masks = _ball_masks(q, n, radius)
    # order: start from a random permutation, then append the rest (deterministic tail)
    base = list(range(N))
    rng.shuffle(base)
    order = base  # covers all words; the tail is just the remainder
    best = None
    # Single full pass (restarting with different orders is bounded by the deadline).
    res = _random_greedy(q, n, radius, masks, order, deadline, rng)
    if res is not None:
        best = res
    return best


def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + (150.0 * _DEADLINE_FRACTION)
    rng = np.random.default_rng(12345)

    baseline = _direct_sum(q, n, radius)

    if n == 0:
        return [[]]

    # Try greedy search only when it is plausibly affordable.
    N = q ** n
    if N <= 200000 and radius >= 1:
        res = _greedy_search(q, n, radius, deadline, rng)
        if res is not None:
            cw = [list(np.frombuffer(np.arange(res[0], dtype=np.uint64).tobytes(),
                                     dtype=np.uint8).reshape(-1, n)[0])]
            # build codewords from indices
            cw = []
            for idx in res:
                w = []
                x = idx
                for _ in range(n):
                    w.append(x % q)
                    x //= q
                cw.append(w[::-1])
            # exact verification before trusting
            if _is_covering(cw, q, n, radius):
                return cw
    return baseline
