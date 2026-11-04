import time
import random
import numpy as np

_cache = {}

def _prep(q, n, radius):
    key = (q, n, radius)
    if key in _cache:
        return _cache[key]
    total = q ** n
    digits = np.zeros((total, n), dtype=np.int64)
    tmp = np.arange(total, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        digits[:, i] = tmp % q
        tmp //= q
    ball = []
    reverse = [[] for _ in range(total)]
    for c in range(total):
        d = digits[c]
        seen = {c}
        for pos in range(n):
            for val in range(q):
                if val == d[pos]:
                    continue
                idx = 0
                for i in range(n):
                    idx = idx * q + (val if i == pos else d[i])
                seen.add(idx)
        arr = np.fromiter(seen, dtype=np.int64)
        ball.append(arr)
        for v in seen:
            reverse[v].append(c)
    _cache[key] = (total, digits, ball, reverse)
    return _cache[key]


def _greedy_init(total, ball, q, n):
    price = np.zeros(total, dtype=np.float64)
    uncovered = np.ones(total, dtype=bool)
    chosen = []
    for _ in range(20000):
        if not uncovered.any():
            break
        best = -1
        best_score = -1.0
        for c in range(total):
            b = ball[c]
            gain = uncovered[b].sum()
            if gain == 0:
                continue
            s = gain - price[c]
            if s > best_score:
                best_score = s
                best = c
        if best < 0:
            break
        chosen.append(best)
        b = ball[best]
        uncovered[b] = False
        price[b] += 1.0
    return chosen


def construct(q, n, radius):
    t0 = time.time()
    deadline = 135.0
    total, digits, ball, reverse = _prep(q, n, radius)

    chosen = _greedy_init(total, ball, q, n)
    chosen_set = set(chosen)

    cover = np.zeros(total, dtype=np.int64)
    for c in chosen_set:
        cover[ball[c]] += 1

    def try_remove(c):
        return bool((cover[ball[c]] >= 2).all())

    changed = True
    while changed:
        changed = False
        for c in list(chosen_set):
            if try_remove(c):
                cover[ball[c]] -= 1
                chosen_set.discard(c)
                changed = True

    for _ in range(500000):
        if time.time() - t0 > deadline:
            break
        if not chosen_set:
            break
        c = random.choice(list(chosen_set))
        b = ball[c]
        cover[b] -= 1
        chosen_set.discard(c)
        newly_uncov = (cover == 0)
        if not newly_uncov.any():
            continue
        # candidate centers: those covering at least one newly uncovered word
        cand_set = set()
        for v in np.where(newly_uncov)[0]:
            cand_set.update(reverse[v])
        best = -1
        best_over = 10**9
        for cand in cand_set:
            if cand in chosen_set:
                continue
            bb = ball[cand]
            if not newly_uncov[bb].all():
                continue
            over = int((cover[bb] >= 1).sum())
            if over < best_over:
                best_over = over
                best = cand
        if best >= 0:
            cover[ball[best]] += 1
            chosen_set.add(best)
        else:
            cover[b] += 1
            chosen_set.add(c)

    changed = True
    while changed:
        changed = False
        for c in list(chosen_set):
            if try_remove(c):
                cover[ball[c]] -= 1
                chosen_set.discard(c)
                changed = True

    return [list(map(int, digits[c])) for c in sorted(chosen_set)]
