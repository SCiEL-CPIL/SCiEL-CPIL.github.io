"""Large 4-uniform sunflower-free family on [1,16].

Constraint: no three distinct sets share a common type (a single element or a
pair of elements).  Equivalently, every element and every pair occurs in at
most two sets.  We build a baseline greedy and grow it with local search.
"""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

DEADLINE = 60.0
N = 16


def _sunflower_free(chosen):
    """Verify the sunflower-free condition (no element/pair in 3+ sets)."""
    elem = [0] * N
    pair = {}
    for m in chosen:
        for x in range(N):
            if (m >> x) & 1:
                elem[x] += 1
                if elem[x] > 2:
                    return False
        bits = [x for x in range(N) if (m >> x) & 1]
        for i in range(4):
            for j in range(i + 1, 4):
                key = bits[i] * N + bits[j]
                nv = pair.get(key, 0) + 1
                pair[key] = nv
                if nv > 2:
                    return False
    return True


def _build():
    sets = []
    for c in combinations(range(N), 4):
        m = 0
        for x in c:
            m |= 1 << x
        sets.append(m)
    nsets = len(sets)
    # deterministic ordering by element-sum then value
    def esum(m):
        s = 0
        for x in range(N):
            if (m >> x) & 1:
                s += x + 1
        return s
    order = list(range(nsets))
    order.sort(key=lambda i: (esum(sets[i]), sets[i]))

    chosen = set()
    elem = [0] * N
    pair = {}

    def try_add(m):
        for x in range(N):
            if (m >> x) & 1 and elem[x] >= 2:
                return False
        bits = [x for x in range(N) if (m >> x) & 1]
        for i2 in range(4):
            for j2 in range(i2 + 1, 4):
                key = bits[i2] * N + bits[j2]
                if pair.get(key, 0) >= 2:
                    return False
        for x in range(N):
            if (m >> x) & 1:
                elem[x] += 1
        for i2 in range(4):
            for j2 in range(i2 + 1, 4):
                key = bits[i2] * N + bits[j2]
                pair[key] = pair.get(key, 0) + 1
        chosen.add(m)
        return True

    def remove(m):
        for x in range(N):
            if (m >> x) & 1:
                elem[x] -= 1
        bits = [x for x in range(N) if (m >> x) & 1]
        for i2 in range(4):
            for j2 in range(i2 + 1, 4):
                key = bits[i2] * N + bits[j2]
                pair[key] -= 1
                if pair[key] == 0:
                    del pair[key]
        chosen.discard(m)

    # ---- Baseline greedy ----
    for i in order:
        if time.time() >= DEADLINE:
            break
        try_add(sets[i])
    best = sorted(chosen)
    best_len = len(best)

    # ---- Local search ----
    improved = True
    while improved and time.time() < DEADLINE:
        improved = False
        # additions
        for i in order:
            if time.time() >= DEADLINE:
                break
            m = sets[i]
            if m in chosen:
                continue
            if try_add(m):
                improved = True
                if len(chosen) > best_len:
                    best_len = len(chosen)
                    best = sorted(chosen)
        # 1-swap moves
        if time.time() < DEADLINE:
            for ra in sorted(chosen):
                if time.time() >= DEADLINE:
                    break
                remove(ra)
                added = False
                for i in order:
                    m = sets[i]
                    if m in chosen:
                        continue
                    if try_add(m):
                        added = True
                        if len(chosen) > best_len:
                            best_len = len(chosen)
                            best = sorted(chosen)
                        break
                if not added:
                    remove(ra) if ra in chosen else None
                    # restore ra
                    try_add(ra)
                else:
                    improved = True
    # final safety re-check
    if not _sunflower_free(best):
        # fall back to a verified greedy
        chosen.clear(); elem = [0]*N; pair = {}
        for i in order:
            try_add(sets[i])
        best = sorted(chosen)
    return best


def construct():
    res = _build()
    if len(res) == 0:
        res = [[1, 2, 3, 4]]
    return res
# EVOLVE-BLOCK-END
