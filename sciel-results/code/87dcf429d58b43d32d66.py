import time
import numpy as np
import random
from collections import defaultdict
import itertools

_TETRA = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]


def _product_code():
    return [h + t for h in _TETRA for t in _TETRA]


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 112.0  # safety margin under 120s

    if q == 3 and n == 8 and radius == 1:
        return _ternary_8_1(deadline)

    # Fallbacks (best-effort) for other parameters
    if q == 3 and n >= 4:
        return [h + list(tail) for h in _TETRA
                for tail in itertools.product(range(3), repeat=n - 4)]
    return [list(c) for c in itertools.product(range(q), repeat=n)]


def _ternary_8_1(deadline):
    n = 8
    q = 3
    total = 3 ** n
    ball_size = 1 + 2 * n  # 17

    weights = np.array([3 ** (n - 1 - i) for i in range(n)], dtype=np.int64)

    # balls[w] = [w, w+/-e_i, w+/-2e_i]
    balls = np.empty((total, ball_size), dtype=np.int64)
    balls[:, 0] = np.arange(total)
    for pos in range(n):
        w = weights[pos]
        digs = (np.arange(total) // w) % 3
        balls[:, 1 + 2 * pos] = np.arange(total) - digs * w + w
        balls[:, 2 + 2 * pos] = np.arange(total) - digs * w + 2 * w

    # For each word, the 17 centers whose ball contains it
    centers = np.empty((total, ball_size), dtype=np.int64)
    for k in range(ball_size):
        centers[balls[:, k], k] = np.arange(total)

    # ---- product-code seed (valid, 486) ----
    seed = np.array(_product_code(), dtype=np.int64)
    cc = np.zeros(total, dtype=np.int64)
    for c in seed:
        cc[balls[c]] += 1
    assert (cc >= 1).all()
    best = seed.copy()
    best_size = len(best)

    # ---- group structure (permutation of coordinates) ----
    perm = np.arange(n)
    perm_inv = np.zeros(n, dtype=np.int64)
    perm_inv[perm] = np.arange(n)
    perm_w = np.zeros(n, dtype=np.int64)
    perm_w[perm] = weights

    def remap(c):
        digs = (c // weights) % 3
        digs[perm_inv] = digs[perm]
        return np.sum(digs * weights)

    def make_orbits():
        orbits = []
        seen = set()
        for c in range(total):
            if c in seen:
                continue
            o = [c]
            seen.add(c)
            for i in range(n):
                for d in (1, 2):
                    nc = remap(c - d * weights[i] + d * perm_w[i])
                    if nc not in seen:
                        seen.add(nc)
                        o.append(nc)
            orbits.append(np.array(o, dtype=np.int64))
        return orbits

    orbits = make_orbits()

    def try_remove(cc, arr, i):
        if (cc[balls[arr[i]]] > 1).all():
            cc[balls[arr[i]]] -= 1
            arr = np.delete(arr, i)
            return arr, True
        return arr, False

    def try_swap(cc, arr, i, c2):
        b1 = balls[arr[i]]
        if (cc[b1] > 1).all():
            cc[b1] -= 1
            b2 = balls[c2]
            cc[b2] += 1
            arr = np.delete(arr, i)
            arr = np.append(arr, c2)
            return arr, True
        uniq = b1[cc[b1] == 1]
        b2 = balls[c2]
        if cc[b2][uniq].min() < 1:
            return arr, False
        cc[b1] -= 1
        cc[b2] += 1
        arr = np.delete(arr, i)
        arr = np.append(arr, c2)
        return arr, True

    def try_add(cc, arr, c2):
        cc[balls[c2]] += 1
        return np.append(arr, c2)

    def greedy_fill(cc, arr):
        while (cc < 1).any() and time.time() < deadline - 12.0:
            cand = np.where((cc < 1)[balls].sum(axis=1) > 0)[0]
            if len(cand) == 0:
                break
            sc = (cc < 1)[balls[cand]].sum(axis=1)
            order = np.argsort(-sc)
            moved = False
            for c in cand[order]:
                arr, ok = try_add(cc, arr, c)
                if ok:
                    moved = True
                    break
            if not moved:
                break
        return arr

    # ---- structured local search from product seed ----
    for restart in range(60):
        if time.time() > deadline - 4.0:
            break
        random.shuffle(perm.tolist())
        p = perm.tolist()
        perm_inv[p] = np.arange(n)
        perm_w[p] = weights

        arr = best.copy()
        cc = np.zeros(total, dtype=np.int64)
        for c in arr:
            cc[balls[c]] += 1

        # pass 1: removals
        for i in range(len(arr) - 1, -1, -1):
            arr, _ = try_remove(cc, arr, i)

        # pass 2: structured swaps
        orbits = make_orbits()
        for it in range(2000):
            if time.time() > deadline - 12.0:
                break
            i = random.randrange(len(arr))
            c1 = arr[i]
            b1 = balls[c1]
            if (cc[b1] > 1).all():
                arr, _ = try_remove(cc, arr, i)
                continue
            uniq = b1[cc[b1] == 1]
            o = orbits[uniq[0]]
            cands = [c for c in o if c not in set(arr.tolist()) and c != c1]
            if not cands:
                continue
            bestc = None
            bestscore = 0
            for c2 in cands:
                if cc[balls[c2]][uniq].min() < 1:
                    continue
                score = int((cc[balls[c2]] == 1).sum())
                if score > bestscore:
                    bestscore = score
                    bestc = c2
            if bestc is not None:
                arr, _ = try_swap(cc, arr, i, bestc)

        # pass 3: removals again
        for i in range(len(arr) - 1, -1, -1, ):
            arr, _ = try_remove(cc, arr, i)

        # safety: fill if somehow invalid
        if (cc < 1).any():
            arr = greedy_fill(cc, arr)

        if len(arr) < best_size:
            best_size = len(arr)
            best = arr.copy()

    # ---- final global removal sweep on the best ----
    cc = np.zeros(total, dtype=np.int64)
    for c in best:
        cc[balls[c]] += 1
    changed = True
    while changed and time.time() < deadline - 1.0:
        changed = False
        for i in range(len(best) - 1, -1, -1):
            if (cc[balls[best[i]]] > 1).all():
                cc[balls[best[i]]] -= 1
                best = np.delete(best, i)
                changed = True

    result = []
    for c in best:
        digs = (c // weights) % 3
        result.append([int(x) for x in digs])
    return result
