"""Rewritten: greedy extraction of edge-disjoint Hamiltonian cycles with pruning."""
import time
import random

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

_N = P['n']
_EDGES = P['edges']
_DEADLINE = time.time() + 80.0  # wall-clock guard; evaluator kills at 90s


def _as_int(v):
    if isinstance(v, bool) or not isinstance(v, int):
        raise TypeError
    return v


def _edge_key(a, b):
    a = _as_int(a)
    b = _as_int(b)
    return (a, b) if a < b else (b, a)


def _build_adjacency(remaining):
    adj = {v: set() for v in range(_N)}
    for a, b in remaining:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def _find_cycle(adj, start, deadline):
    n = _N
    full = set(range(n))
    target = adj[start]

    def dfs(cur, path, used):
        if time.time() > deadline:
            return None
        if len(path) == n:
            if cur in target:
                return path
            return None
        nbrs = list(adj[cur])
        nbrs.sort(key=lambda w: (len(adj[w]), w))
        for w in nbrs:
            if w in used:
                continue
            if len(path) == n - 1 and w != start:
                continue
            used.add(w)
            path.append(w)
            res = dfs(w, path, used)
            if res is not None:
                return res
            path.pop()
            used.discard(w)
        return None

    return dfs(start, [start], {start})


def _try_one(remaining, adj, deadline):
    starts = sorted(range(_N), key=lambda v: (len(adj[v]), v))
    for s in starts:
        if time.time() > deadline:
            break
        c = _find_cycle(adj, s, deadline)
        if c is not None:
            return c
    return None


def _extract_cycles():
    cycles = []
    remaining = set(_edge_key(a, b) for a, b in _EDGES)
    adj = _build_adjacency(remaining)
    for _ in range(10):
        if time.time() > _DEADLINE:
            break
        c = _try_one(remaining, adj, _DEADLINE)
        if c is None:
            break
        cycles.append(c)
        for i in range(len(c)):
            remaining.discard(_edge_key(c[i], c[(i + 1) % len(c)]))
        adj = _build_adjacency(remaining)
    return cycles


def _normalize(cycles):
    out = []
    for c in cycles:
        out.append([_as_int(v) for v in c])
    return out


def _valid(cycles):
    if not cycles:
        return False
    seen = None
    used = set()
    for c in cycles:
        if not isinstance(c, (list, tuple)) or len(c) < 3:
            return False
        vs = [_as_int(v) for v in c]
        if len(set(vs)) != len(vs):
            return False
        s = frozenset(vs)
        if seen is None:
            seen = s
        elif s != seen:
            return False
        for i in range(len(vs)):
            k = _edge_key(vs[i], vs[(i + 1) % len(vs)])
            if k in used:
                return False
            used.add(k)
    return True


def construct():
    random.seed(12345)
    try:
        cycles = _extract_cycles()
        if not _valid(cycles):
            cycles = []
    except Exception:
        cycles = []
    if not cycles:
        # guaranteed-valid fallback: one Hamiltonian cycle over all 21 vertices
        cycles = [[v for v in range(_N)]]
    return _normalize(cycles)
# EVOLVE-BLOCK-END
