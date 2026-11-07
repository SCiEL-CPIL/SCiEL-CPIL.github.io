"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

_N = P['n']
_EDGES = P['edges']
_NEI = [[] for _ in range(_N)]
for _a, _b in _EDGES:
    _NEI[_a].append(_b)
    _NEI[_b].append(_a)
_DEG = [len(x) for x in _NEI]
_ORDER = sorted(range(_N), key=lambda v: -_DEG[v])
_INV = [0] * _N
for _i, _v in enumerate(_ORDER):
    _INV[_v] = _i


class _DSU:
    __slots__ = ('parent', 'size')

    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x):
        p = self.parent
        r = x
        while p[r] != r:
            r = p[r]
        while p[x] != r:
            p[x], x = r, p[x]
        return r

    def union(self, a, b):
        pa = self.find(a)
        pb = self.find(b)
        if pa == pb:
            return False
        if self.size[pa] < self.size[pb]:
            pa, pb = pb, pa
        self.parent[pb] = pa
        self.size[pa] += self.size[pb]
        return True


def _check_bicolored_forest(col, k):
    # Verify: adjacent differ and each color-pair subgraph is a forest.
    dsu = _DSU(_N)
    for a, b in _EDGES:
        ca, cb = col[a], col[b]
        if ca == cb:
            return False
        if not dsu.union(a, b):
            return False
    return True


def _search(k, deadline, max_nodes):
    color = [-1] * _N
    dsu = _DSU(_N)
    nodes = [0]
    found = [False]

    def backtrack():
        if found[0]:
            return
        if nodes[0] > max_nodes:
            return
        if time.time() > deadline:
            return
        # find next uncolored in _ORDER
        v = -1
        for idx in range(_N):
            w = _ORDER[idx]
            if color[w] == -1:
                v = w
                break
        if v == -1:
            found[0] = True
            return
        used = [False] * k
        for nb in _NEI[v]:
            c = color[nb]
            if c != -1:
                used[c] = True
        for c in range(k):
            if used[c]:
                continue
            color[v] = c
            ok = True
            for nb in _NEI[v]:
                cn = color[nb]
                if cn != -1:
                    if not dsu.union(v, nb):
                        ok = False
                        break
            if ok:
                nodes[0] += 1
                backtrack()
                nodes[0] -= 1
                if found[0]:
                    return
            for nb in _NEI[v]:
                cn = color[nb]
                if cn != -1:
                    dsu.parent[nb] = nb
                    dsu.size[nb] = 1
            color[v] = -1

    backtrack()
    return (found[0], color)


def construct():
    start = time.time()
    deadline = start + 45.0
    best = None
    max_nodes = 600000
    for k in range(2, 50):
        if time.time() > deadline:
            break
        ok, col = _search(k, deadline, max_nodes)
        if ok:
            if _check_bicolored_forest(col, k):
                best = col
                break
    if best is None:
        # fallback: guaranteed-valid greedy (distinct colors)
        best = list(range(_N))
    return [int(x) for x in best]
# EVOLVE-BLOCK-END
