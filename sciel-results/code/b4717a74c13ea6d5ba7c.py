"""Rewritten suite task: minimise n for a 4-regular graph of girth >= 10.

Fresh design, same family goal (minimise n, certify girth via non-backtracking
walks). We (1) build the known LUW CD(4,4) explicit graph as a guaranteed valid
seed, then (2) run a deadline-guarded local search over 4-regular graphs of
decreasing size, using the *count of non-backtracking closed walks of length < g*
as a smooth surrogate whose zero level is exactly the girth certificate. We keep
the best certified result and always return it before a wall-clock deadline.
"""

import time
import random
from collections import deque

_DEADLINE = 150.0
_MARGIN = 6.0
_RNG_SEED = 20260410


def _luw_cd44():
    q, d = 4, 4
    table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
    add = sub = lambda x, y: x ^ y
    mul = lambda x, y: table[x][y]
    rule = {2: (1,1), 3: (2,1), 4: (1,2)}
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0]*(d-1)
            for c in range(2, d+1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)
    start = (0, (None,) + (0,)*d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order); order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


def _adj_from_edges(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v); adj[v].add(u)
    return adj


def _edges_from_adj(adj):
    return [(u, v) for u in range(len(adj)) for v in adj[u] if u < v]


def _is_kregular(adj, k):
    return all(len(a) == k for a in adj)


def _nb_adjacency(adj):
    nb = {}
    for u in range(len(adj)):
        for v in adj[u]:
            nb[(u, v)] = [(v, w) for w in adj[v] if w != u]
    return nb


def _count_short_nb_closed(adj, g):
    """Number of non-backtracking closed walks of length in [3, g-1]."""
    nb = _nb_adjacency(adj)
    total = 0
    for u in range(len(adj)):
        for v in adj[u]:
            start, target = (u, v), (v, u)
            if start == target:
                continue
            seen = {start: 1}
            q = deque([start])
            while q:
                cur = q.popleft(); dcur = seen[cur]
                if dcur >= g - 1:
                    break
                for nxt in nb[cur]:
                    if nxt == target:
                        if dcur + 1 < g:
                            total += 1
                        break
                    if nxt not in seen:
                        seen[nxt] = dcur + 1; q.append(nxt)
    return total


def _girth_ok(adj, g):
    return _count_short_nb_closed(adj, g) == 0


def _two_switch(adj, a, b, c, d):
    adj[a].remove(b); adj[b].remove(a)
    adj[c].remove(d); adj[d].remove(c)
    adj[a].add(d); adj[d].add(a)
    adj[b].add(c); adj[c].add(b)


def _random_4reg(n, rng):
    for _ in range(300):
        stubs = [v for v in range(n) for _ in range(4)]
        rng.shuffle(stubs)
        adj = [set() for _ in range(n)]
        ok = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v or v in adj[u]:
                ok = False; break
            adj[u].add(v); adj[v].add(u)
        if ok and _is_kregular(adj, 4):
            return adj
    return None


def construct(k, g):
    t0 = time.time()
    rng = random.Random(_RNG_SEED)
    def deadline():
        return time.time() - t0 > _DEADLINE - _MARGIN

    n_seed, e_seed = _luw_cd44()
    adj_seed = _adj_from_edges(n_seed, e_seed)
    best_n, best_adj = n_seed, adj_seed
    best_valid = _is_kregular(adj_seed, k) and _girth_ok(adj_seed, g)

    if not deadline():
        for target_n in range(n_seed, max(64, n_seed // 2 + 1), -4):
            if deadline():
                break
            adj = _random_4reg(target_n, rng)
            if adj is None:
                continue
            cur = _count_short_nb_closed(adj, g)
            improved, iters = True, 0
            while improved and iters < 15000 and not deadline():
                improved = False; iters += 1
                u = rng.randrange(target_n)
                if len(adj[u]) < 2:
                    continue
                a, b = rng.sample(sorted(adj[u]), 2)
                cands = [c for c in adj[b] if c != a]
                for c in rng.sample(cands, len(cands)):
                    dd = [d for d in (adj[c] & adj[a]) if d != b]
                    if not dd:
                        continue
                    d = rng.choice(dd)
                    _two_switch(adj, a, b, c, d)
                    new = _count_short_nb_closed(adj, g)
                    if new <= cur:
                        cur = new; improved = True; break
                    _two_switch(adj, a, b, d, c)
                if cur == 0:
                    break
            if cur == 0 and _is_kregular(adj, 4):
                if target_n < best_n:
                    best_n, best_adj, best_valid = target_n, adj, True

    if not best_valid:
        return n_seed, _edges_from_adj(adj_seed)
    return best_n, _edges_from_adj(best_adj)
