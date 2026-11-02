# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import defaultdict

_START = time.time()
_DEADLINE = 150.0
_MARGIN = 3.0


def _time_left():
    return _DEADLINE - (time.time() - _START)


def construct(q, n, radius):
    total = q ** n

    def int_to_word(v):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = v % q
            v //= q
        return w

    # digits of every word
    digits = np.empty((total, n), dtype=np.int32)
    for v in range(total):
        w = int_to_word(v)
        digits[v] = w

    # Build ball membership (bool matrix) total x total
    if radius == 1:
        ball = np.zeros((total, total), dtype=np.bool_)
        idx = np.arange(total)
        ball[idx, idx] = True
        for i in range(n):
            pow_q_i = q ** i
            pow_q_ip1 = q ** (i + 1)
            key = (np.arange(total) // pow_q_ip1) * pow_q_i + (np.arange(total) % pow_q_i)
            groups = defaultdict(list)
            for w in range(total):
                groups[key[w]].append(w)
            for c in range(total):
                for w in groups[key[c]]:
                    ball[c, w] = True
    else:
        diff = (digits[:, None, :] != digits[None, :, :])
        dist = diff.sum(axis=2)
        ball = dist <= radius

    ball_int = ball.astype(np.int32)

    # Greedy
    uncovered = np.ones(total, dtype=np.bool_)
    unc_int = uncovered.astype(np.int32)
    chosen = []

    while uncovered.any() and _time_left() > _MARGIN:
        counts = ball_int @ unc_int
        best = int(np.argmax(counts))
        if counts[best] == 0:
            break
        chosen.append(best)
        uncovered[ball[best]] = False
        unc_int[ball[best]] = 0

    def try_remove(chosen):
        if not chosen:
            return chosen
        cov = np.zeros(total, dtype=np.int32)
        for c in chosen:
            cov[ball[c]] += 1
        new_chosen = []
        for c in chosen:
            only_c = ball[c] & (cov == 1)
            if only_c.any():
                new_chosen.append(c)
            else:
                cov[ball[c]] -= 1
        return new_chosen

    chosen = try_remove(chosen)

    # Limited swap improvement
    if _time_left() > _MARGIN + 5:
        chosen_set = set(chosen)
        for _ in range(3):  # a few rounds
            if _time_left() <= _MARGIN + 5:
                break
            cov = np.zeros(total, dtype=np.int32)
            for c in chosen:
                cov[ball[c]] += 1
            improved = False
            for c in list(chosen):
                critical = ball[c] & (cov == 1)
                if not critical.any():
                    chosen.remove(c)
                    chosen_set.discard(c)
                    improved = True
                    break
                crit_count = int(critical.sum())
                if crit_count == 0:
                    continue
                covers = (ball_int @ critical.astype(np.int32)) == crit_count
                for d in range(total):
                    if d in chosen_set:
                        continue
                    if covers[d]:
                        chosen.remove(c)
                        chosen.append(d)
                        chosen_set.discard(c)
                        chosen_set.add(d)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                chosen = try_remove(chosen)

    chosen = try_remove(chosen)

    result = [int_to_word(c) for c in chosen]
    seen = set()
    out = []
    for w in result:
        t = tuple(w)
        if t not in seen:
            seen.add(t)
            out.append(w)
    return out
# EVOLVE-BLOCK-END
