"""Few colors without any bicolored cycle - minimize distinct colors (target 3)."""
import random
import time

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
EDGES = P['edges']

def _pair_edges(coloring):
    d = {}
    for a, b in EDGES:
        ca, cb = coloring[a], coloring[b]
        if ca == cb:
            continue
        key = (ca, cb) if ca < cb else (cb, ca)
        d.setdefault(key, []).append((a, b))
    return d

def _has_cycle(edges):
    if not edges:
        return False
    parent = {}
    def find(x):
        r = x
        while parent.get(r, r) != r:
            r = parent.get(r, r)
        while parent.get(x, x) != r:
            parent[x], x = r, parent.get(x, x)
        return r
    for u, v in edges:
        ru, rv = find(u), find(v)
        if ru == rv:
            return True
        parent[ru] = rv
    return False

def _bad(coloring):
    b = 0
    for a, bb in EDGES:
        if coloring[a] == coloring[bb]:
            b += 1
    for key, el in _pair_edges(coloring).items():
        if _has_cycle(el):
            b += 1
    return b

def _anneal(coloring, k, rng, deadline):
    cur = list(coloring)
    cur_bad = _bad(cur)
    if cur_bad == 0:
        return cur
    T = 4.0
    cool = 0.9997
    it = 0
    while it < 80000 and time.time() < deadline:
        it += 1
        v = rng.randrange(N)
        old = cur[v]
        new = rng.randrange(k)
        if new == old:
            continue
        cur[v] = new
        nb = _bad(cur)
        if nb <= cur_bad:
            cur_bad = nb
            if nb == 0:
                return cur
        else:
            acc = (nb - cur_bad) / (T + 1e-9)
            if rng.random() < min(1.0, 0.6 * acc):
                cur_bad = nb
            else:
                cur[v] = old
        T *= cool
    return cur

def construct():
    deadline = time.time() + 80.0
    rng = random.Random(20240501)
    best = None
    best_bad = None

    for k in (3, 4, 5):
        if time.time() > deadline:
            break
        starts = []
        for mod in (3, 4, 5, 6, 7, 10):
            starts.append([(i % mod) % k for i in range(N)])
        for _ in range(60):
            starts.append([rng.randrange(k) for _ in range(N)])

        for s in starts:
            if time.time() > deadline:
                break
            c = _anneal(s, k, rng, deadline)
            b = _bad(c)
            if b == 0:
                return [int(x) for x in c]
            if best_bad is None or b < best_bad:
                best_bad = b
                best = list(c)

    if best is not None:
        c = _anneal(best, 5, rng, deadline)
        if _bad(c) == 0:
            return [int(x) for x in c]
        # Guaranteed-valid fallback: distinct color per vertex (no bicolored edges -> forests).
        return [int(i) for i in range(N)]

    return [int(i) for i in range(N)]

if __name__ == "__main__":
    r = construct()
    print("distinct:", len(set(r)), "valid:", _bad(r) == 0)
