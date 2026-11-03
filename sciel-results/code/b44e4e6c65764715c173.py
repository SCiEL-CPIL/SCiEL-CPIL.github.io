"""Vectorized greedy + tetracode structured init + targeted local search
for K_q(n, radius) covering codes (football pool / ternary covering)."""
import time
import itertools
import numpy as np

_DEADLINE = 150.0


def _digits_to_int(digits, base):
    num = 0
    for d in reversed(digits):
        num = num * base + d
    return num


def _int_to_digits(num, length, base):
    digits = [0] * length
    for i in range(length):
        digits[i] = num % base
        num //= base
    return digits


def _tetracode_words(q, n):
    """Deterministic structured seed: tetracode (q=3, len 4, radius 1) product.
    Returns a list of integer codeword indices that already cover a lot."""
    if q != 3 or n < 4:
        return []
    tetracode = [
        [0, 0, 0, 0],
        [0, 1, 2, 1],
        [0, 2, 1, 2],
        [1, 0, 1, 2],
        [1, 1, 2, 0],
        [1, 2, 0, 2],
    ]
    out = []
    seen = set()
    if n == 4:
        for w in tetracode:
            idx = _digits_to_int(w, q)
            if idx not in seen:
                seen.add(idx)
                out.append(idx)
        return out
    # product of two tetracodes on first 8 positions, then pad
    for a in tetracode:
        for b in tetracode:
            head = a + b
            if n > 8:
                head = head + [0] * (n - 8)
            idx = _digits_to_int(head[:n], q)
            if idx not in seen:
                seen.add(idx)
                out.append(idx)
    return out


def construct(q, n, radius):
    t_start = time.time()
    deadline = t_start + _DEADLINE - 8.0  # safety margin
    total = q ** n

    # ---- Precompute the ball matrix: balls[i] = list of words within radius ----
    ball_list = [None] * total
    if radius == 1:
        for i in range(total):
            d = _int_to_digits(i, n, q)
            nbrs = [i]
            for pos in range(n):
                off = q ** (n - 1 - pos)
                cur = d[pos]
                for v in range(q):
                    if v != cur:
                        nbrs.append(i + (v - cur) * off)
            ball_list[i] = nbrs
        ball_size = 1 + n * (q - 1)
    else:
        for i in range(total):
            d = _int_to_digits(i, n, q)
            nbrs = [i]
            for j in range(total):
                if j == i:
                    continue
                d2 = _int_to_digits(j, n, q)
                dist = sum(1 for a, b in zip(d, d2) if a != b)
                if dist <= radius:
                    nbrs.append(j)
            ball_list[i] = nbrs
        ball_size = max(len(x) for x in ball_list)

    # Build a dense matrix with -1 padding for vectorized boolean indexing
    balls = np.full((total, ball_size), -1, dtype=np.int32)
    for i in range(total):
        balls[i, :len(ball_list[i])] = ball_list[i]

    # Inverse index: for each word w, the set of centers whose ball contains w.
    # Enables O(1) candidate enumeration for targeted swaps.
    centers_covering = [None] * total
    for i in range(total):
        for w in ball_list[i]:
            if centers_covering[w] is None:
                centers_covering[w] = []
            centers_covering[w].append(i)
    for w in range(total):
        if centers_covering[w] is not None:
            centers_covering[w] = frozenset(centers_covering[w])

    def _greedy_extend(code, cover_count, seed):
        """Vectorized greedy: repeatedly add the center with max uncovered gain."""
        rng = np.random.RandomState(seed)
        code = list(code)
        code_set = set(code)
        uncovered = (cover_count == 0)
        while np.any(uncovered):
            if time.time() > deadline:
                break
            gains = np.zeros(total, dtype=np.int32)
            gains[balls[uncovered]] += 1  # count uncovered words each center covers
            gains[list(code_set)] = -1
            max_gain = gains.max()
            if max_gain <= 0:
                break
            best = np.where(gains == max_gain)[0]
            if len(best) == 1:
                c = int(best[0])
            else:
                c = int(best[rng.randint(len(best))])
            code.append(c)
            code_set.add(c)
            cover_count[balls[c]] += 1
            uncovered = (cover_count == 0)
        return code, cover_count

    def _trim(code, cover_count):
        """Remove any codeword whose ball is fully double-covered."""
        changed = True
        while changed and time.time() < deadline:
            changed = False
            to_remove = []
            for idx, c in enumerate(code):
                if np.any(cover_count[balls[c]] == 1):
                    continue
                to_remove.append(idx)
            if to_remove:
                for idx in sorted(to_remove, reverse=True):
                    c = code[idx]
                    code.pop(idx)
                    cover_count[balls[c]] -= 1
                changed = True
        return code, cover_count

    def _local_search(code, cover_count, seed):
        """Redundant removal + targeted swap (covers a codeword's uniquely-covered
        words with a single new center), with occasional 2-swap escape moves."""
        rng = np.random.RandomState(seed)
        code = list(code)
        code_set = set(code)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            order = list(range(len(code)))
            rng.shuffle(order)
            for ci in order:
                if time.time() > deadline:
                    break
                i = code[ci]
                i_ball = balls[i]
                only = i_ball[cover_count[i_ball] == 1]
                if len(only) == 0:
                    code.pop(ci)
                    cover_count[i_ball] -= 1
                    code_set.discard(i)
                    improved = True
                    break
                # Candidate centers covering every uniquely-covered word of i
                cand = None
                for w in only:
                    if cand is None:
                        cand = set(centers_covering[w])
                    else:
                        cand &= centers_covering[w]
                    if not cand:
                        break
                if cand:
                    cand -= code_set
                    best_add = None
                    best_gain = -1
                    for a in list(cand)[:64]:
                        a_ball = balls[a]
                        gain = int(np.isin(only, a_ball).sum())
                        if gain > best_gain:
                            best_gain = gain
                            best_add = a
                        if gain == len(only):
                            break
                    if best_add is not None and best_gain == len(only):
                        cover_count[balls[i]] -= 1
                        cover_count[balls[best_add]] += 1
                        code[ci] = best_add
                        code_set.discard(i)
                        code_set.add(best_add)
                        improved = True
                        break
            # Escape move: accept a swap that keeps size but reshapes coverage
            if not improved and len(code) > 2 and time.time() < deadline:
                if rng.random() < 0.3:
                    ci = rng.randint(len(code))
                    i = code[ci]
                    i_ball = balls[i]
                    only = i_ball[cover_count[i_ball] == 1]
                    if len(only) > 0:
                        cand = None
                        for w in only:
                            if cand is None:
                                cand = set(centers_covering[w])
                            else:
                                cand &= centers_covering[w]
                            if not cand:
                                break
                        if cand:
                            cand -= code_set
                            if cand:
                                a = list(cand)[rng.randint(len(cand))]
                                a_ball = balls[a]
                                # ensure swap does not leave anything uncovered
                                if np.all(cover_count[i_ball] >= 1 or np.isin(i_ball, a_ball)):
                                    cover_count[i_ball] -= 1
                                    cover_count[a_ball] += 1
                                    code[ci] = a
                                    code_set.discard(i)
                                    code_set.add(a)
                                    improved = True
        return code, cover_count

    def _valid(code):
        cc = np.zeros(total, dtype=np.int32)
        for c in code:
            cc[balls[c]] += 1
        return np.all(cc >= 1)

    # ---- Multi-start: several structured / randomized starts, keep the best ----
    best = None
    best_len = total + 1
    starts = []
    # start 0: tetracode structured
    starts.append(('tetracode', _tetracode_words(q, n)))
    # a couple of random starts (small random seed set) for diversity
    for s in (1, 7, 13):
        rng = np.random.RandomState(s)
        k = max(1, total // 200)
        starts.append(('rand%d' % s, list(rng.choice(total, size=k, replace=False))))

    for name, seed_code in starts:
        if time.time() > deadline - 3.0:
            break
        cover_count = np.zeros(total, dtype=np.int32)
        code = list(seed_code)
        # dedupe
        code = list(dict.fromkeys(code))
        for c in code:
            cover_count[balls[c]] += 1
        # extend to full coverage
        code, cover_count = _greedy_extend(code, cover_count, seed=hash(name) % (2**31))
        # trim + local search
        code, cover_count = _trim(code, cover_count)
        code, cover_count = _local_search(code, cover_count, seed=hash(name) % (2**31))
        # final trim
        code, cover_count = _trim(code, cover_count)
        if not _valid(code):
            continue
        if len(code) < best_len:
            best_len = len(code)
            best = code

    # Fallback if nothing valid (shouldn't happen)
    if best is None:
        cover_count = np.zeros(total, dtype=np.int32)
        code, cover_count = _greedy_extend([], cover_count, seed=0)
        code, cover_count = _trim(code, cover_count)
        if not _valid(code):
            code = list(range(total))
        best = code

    # Convert integer indices back to list of words
    return [_int_to_digits(idx, n, q) for idx in best]
