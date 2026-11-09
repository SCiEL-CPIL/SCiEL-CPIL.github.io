"""Minimize distinct colors for degree-4 graph with bicolored-forest constraint.
Uses incremental union-find with rollback for efficient backtracking."""
import time
import random
from collections import deque

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

_N = P['n']
_EDGES = P['edges']
_ADJ = [[] for _ in range(_N)]
for _a, _b in _EDGES:
    _ADJ[_a].append(_b)
    _ADJ[_b].append(_a)

# Edge index lookup: (u,v) -> edge index (for u<v)
_EDGE_IDX = {}
for _i, (_a, _b) in enumerate(_EDGES):
    if _a > _b:
        _a, _b = _b, _a
    _EDGE_IDX[(_a, _b)] = _i


def _check_bicforest(coloring):
    """Check bicolored-forest constraint for a complete coloring."""
    from collections import defaultdict
    pair_edges = defaultdict(list)
    for ei, (a, b) in enumerate(_EDGES):
        ca, cb = coloring[a], coloring[b]
        if ca == cb:
            return False
        cmin, cmax = (ca, cb) if ca < cb else (cb, ca)
        pair_edges[(cmin, cmax)].append(ei)

    for pair, eis in pair_edges.items():
        parent = list(range(_N))
        rank = [0] * _N

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for ei in eis:
            a, b = _EDGES[ei]
            ra, rb = find(a), find(b)
            if ra == rb:
                return False
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1
    return True


class _RollbackUF:
    """Union-find with rollback support (no path compression, use rank)."""
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.history = []  # stack of (child, parent, rank_changed)

    def find(self, x):
        while self.parent[x] != x:
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        # rx has higher or equal rank, attach ry to rx
        self.history.append((ry, rx, self.rank[rx] == self.rank[ry]))
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True

    def snapshot(self):
        return len(self.history)

    def rollback(self, snap):
        while len(self.history) > snap:
            child, parent, rank_changed = self.history.pop()
            self.parent[child] = child
            if rank_changed:
                self.rank[parent] -= 1


def _backtrack_rb(k, deadline):
    """Backtracking with rollback UF for each color pair."""
    color = [-1] * _N
    n_pairs = k * (k - 1) // 2

    def pair_idx(i, j):
        if i > j:
            i, j = j, i
        return i * k - i * (i + 1) // 2 + (j - i - 1)

    ufs = [_RollbackUF(_N) for _ in range(n_pairs)]

    # Vertex ordering: try most constrained first
    vertex_order = sorted(range(_N), key=lambda v: (-len(_ADJ[v]), v))

    nodes_visited = [0]

    def backtrack(pos):
        if time.time() > deadline:
            return False
        nodes_visited[0] += 1
        if pos == _N:
            return True

        v = vertex_order[pos]
        used = set()
        for nb in _ADJ[v]:
            if color[nb] != -1:
                used.add(color[nb])

        # Try colors in order of least conflict
        for c in range(k):
            if c in used:
                continue

            color[v] = c
            snaps = []
            ok = True

            for u in _ADJ[v]:
                cu = color[u]
                if cu == -1 or cu == c:
                    continue
                ci, cj = (c, cu) if c < cu else (cu, c)
                pi = pair_idx(ci, cj)
                snap = ufs[pi].snapshot()
                snaps.append((pi, snap))

                # Find edge index between v and u
                va, vb = (v, u) if v < u else (u, v)
                ei = _EDGE_IDX.get((va, vb))
                if ei is not None:
                    if not ufs[pi].union(va, vb):
                        ok = False
                        break

            if ok:
                if backtrack(pos + 1):
                    # Rollback before returning
                    for pi, snap in reversed(snaps):
                        ufs[pi].rollback(snap)
                    return True

            # Rollback
            for pi, snap in reversed(snaps):
                ufs[pi].rollback(snap)
            color[v] = -1

        return False

    if backtrack(0):
        return color[:]
    return None


def _try_k(k, deadline, num_restarts=20):
    """Try to find valid k-coloring."""
    # Quick greedy check
    order = sorted(range(_N), key=lambda v: (-len(_ADJ[v]), v))
    color = [-1] * _N
    for v in order:
        used = set(color[nb] for nb in _ADJ[v] if color[nb] != -1)
        for c in range(k):
            if c not in used:
                color[v] = c
                break
        if color[v] == -1:
            break
    if color[v] != -1 and _check_bicforest(color):
        return color

    # Randomized greedy
    rng = random.Random(42 + k)
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        perm = list(range(_N))
        rng.shuffle(perm)
        # Sort by degree descending with random tiebreak
        perm.sort(key=lambda v: (-len(_ADJ[v]), rng.random()))
        color = [-1] * _N
        ok = True
        for v in perm:
            used = set(color[nb] for nb in _ADJ[v] if color[nb] != -1)
            avail = [c for c in range(k) if c not in used]
            if not avail:
                ok = False
                break
            color[v] = rng.choice(avail)
        if ok and _check_bicforest(color):
            return color

    # Backtracking with rollback
    if k <= 5:
        bt = _backtrack_rb(k, deadline)
        if bt is not None:
            return bt

    return None


def construct():
    t_start = time.time()
    deadline = t_start + 82.0

    # Check 2-coloring
    color2 = [-1] * _N
    bipartite = True
    for start in range(_N):
        if color2[start] != -1:
            continue
        color2[start] = 0
        q = deque([start])
        while q and bipartite:
            u = q.popleft()
            cu = color2[u]
            for v in _ADJ[u]:
                if color2[v] == -1:
                    color2[v] = 1 - cu
                    q.append(v)
                elif color2[v] == cu:
                    bipartite = False
                    break

    if bipartite and _check_bicforest(color2):
        return [int(x) for x in color2]

    # Try k=3 first with most effort
    for k in range(3, 10):
        if time.time() > deadline - 3.0:
            break
        result = _try_k(k, deadline - 3.0, num_restarts=30 if k <= 3 else 10)
        if result is not None:
            return [int(x) for x in result]

    # Fallback greedy
    for k in range(5, 16):
        if time.time() > deadline - 2.0:
            break
        order = sorted(range(_N), key=lambda v: (-len(_ADJ[v]), v))
        color = [-1] * _N
        valid = True
        for v in order:
            used = set(color[nb] for nb in _ADJ[v] if color[nb] != -1)
            for c in range(k):
                if c not in used:
                    color[v] = c
                    break
            if color[v] == -1:
                valid = False
                break
        if valid and _check_bicforest(color):
            return [int(x) for x in color]

    return [int(i) for i in range(_N)]
