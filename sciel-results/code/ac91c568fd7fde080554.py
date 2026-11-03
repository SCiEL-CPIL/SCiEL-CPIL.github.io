"""Covering code K_3(8,1): greedy head-selection over the tetracode direct-sum family."""
import time
import itertools

_TETRA = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
_TETRA_SET = frozenset(tuple(t) for t in _TETRA)
_TAIL_IDS = list(range(3 ** 4))


def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin under the 150 s wall-clock kill

    # Trivial exact cases (small spaces are easy to cover perfectly or trivially).
    if q ** n <= 1 + n * (q - 1):
        # A single word covers everything.
        return [[0] * n]
    if q == 2 and n == 1:
        return [[0], [1]]
    if q == 3 and n == 1:
        return [[0], [1], [2]]

    if q != 3 or n != 8 or radius != 1:
        # Generic fallback: exhaustive small-space, else the seed construction.
        if q ** n <= 200000:
            words = [list(w) for w in itertools.product(range(q), repeat=n)]
            chosen = []
            covered = set()
            for w in words:
                if tuple(w) in covered:
                    continue
                chosen.append(w)
                for i in range(n):
                    for v in range(q):
                        if v == w[i]:
                            covered.add(tuple(w))
                        else:
                            nw = w[:i] + [v] + w[i + 1:]
                            covered.add(tuple(nw))
                if len(covered) >= q ** n:
                    break
            return chosen
        # Direct-sum seed fallback for other parameters.
        base = n - 4
        if base >= 0:
            return [h + list(t) for h in _TETRA
                    for t in itertools.product(range(q), repeat=base)]
        return [list(t) for t in itertools.product(range(q), repeat=n)]

    # ---- Specialized K_3(8,1) via tetracode direct sum + greedy head selection ----
    heads = _TETRA
    H = len(heads)  # 9
    T = len(_TAIL_IDS)  # 81

    # Precompute, for each tail, which heads cover it (head == tail[:4] or one diff).
    tail_head = []
    for tid in _TAIL_IDS:
        t = [tid // (3 ** (3 - i)) % 3 for i in range(4)]
        hset = set()
        for hi, h in enumerate(heads):
            if h == t:
                hset.add(hi)
                continue
            d = 0
            for k in range(4):
                if h[k] != t[k]:
                    d += 1
                    if d > 1:
                        break
            if d <= 1:
                hset.add(hi)
        tail_head.append(hset)

    # For each head, the set of tails it covers (tail in hset).
    head_covers = []
    for hi in range(H):
        s = set()
        for tid in _TAIL_IDS:
            if hi in tail_head[tid]:
                s.add(tid)
        head_covers.append(s)

    # A codeword (head hi, tail tid) covers:
    #   - all heads in tail_head[tid] at the SAME tail tid  (head varies, tail fixed)
    #   - head hi at every tail in head_covers[hi]          (tail varies, head fixed)
    # Represent a codeword's covered (head,tail) cells as a bitmask: head*81 + tail.
    cell = [hi * T + tid for hi in range(H) for tid in _TAIL_IDS]
    # Precompute codeword bitmasks.
    cw_mask = [0] * (H * T)
    for hi in range(H):
        hc = head_covers[hi]
        base_hi = hi * T
        for tid in _TAIL_IDS:
            m = 0
            # same tail, all covering heads
            for h2 in tail_head[tid]:
                m |= 1 << (h2 * T + tid)
            # same head, all covering tails
            for t2 in hc:
                m |= 1 << (base_hi + t2)
            cw_mask[hi * T + tid] = m

    # Greedy set cover.
    uncovered = (1 << (H * T)) - 1
    chosen = []
    chosen_cells = [False] * (H * T)
    total_cells = H * T  # 729

    while uncovered:
        if time.time() > deadline:
            break
        best = -1
        best_gain = 0
        for idx in range(H * T):
            gain = (cw_mask[idx] & uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best = idx
                if gain == 0:
                    break
        if best < 0:
            break
        chosen.append(best)
        chosen_cells[best // T] = True
        uncovered &= ~cw_mask[best]

    # Convert chosen cell indices back to codewords.
    result = []
    for idx in chosen:
        hi = idx // T
        tid = idx % T
        t = [tid // (3 ** (3 - i)) % 3 for i in range(4)]
        result.append(heads[hi] + t)

    # Safety: if greedy was cut off and did not cover everything, fall back to the
    # guaranteed (larger) full direct-sum construction.
    if uncovered:
        return [h + list(t) for h in heads
                for t in itertools.product(range(3), repeat=4)]
    return result
