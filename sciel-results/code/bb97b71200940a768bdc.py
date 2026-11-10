"""Rewritten: branch-and-bound search for a large clique subdivision."""
# EVOLVE-BLOCK-START
import time as _t
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

_N = P['n']
_EDGES = P['edges']
_DEADLINE = _t.time() + 80.0
_ADJ = [set() for _ in range(_N)]
for _a, _b in _EDGES:
    _ADJ[_a].add(_b)
    _ADJ[_b].add(_a)
_ADJ_LIST = [list(s) for s in _ADJ]
_DEG = [len(s) for s in _ADJ]
# order branches by descending degree (likely hubs)
_ORDER = sorted(range(_N), key=lambda x: (-_DEG[x], x))


def _shortest_path(s, t, banned):
    """BFS shortest simple path s->t avoiding banned. Returns list or None."""
    if s == t:
        return [s]
    if t in banned:
        return None
    from collections import deque
    prev = {s: None}
    q = deque([s])
    while q:
        u = q.popleft()
        for w in _ADJ_LIST[u]:
            if w in prev or w in banned:
                continue
            prev[w] = u
            if w == t:
                path = [t]
                cur = t
                while cur != s:
                    cur = prev[cur]
                    path.append(cur)
                path.reverse()
                return path
            q.append(w)
    return None


def _try_complete(branches, banned, budget):
    """Greedy: route paths for every unordered pair of branches avoiding banned.
    Returns (paths_dict, banned) on success else (None, None)."""
    n = len(branches)
    paths = {}
    banned = set(banned)
    for i in range(n):
        for j in range(i + 1, n):
            a = branches[i]; b = branches[j]
            if _t.time() > budget:
                return None, None
            p = _shortest_path(a, b, banned)
            if p is None:
                return None, None
            paths[(a, b)] = p
            for v in p[1:-1]:
                banned.add(v)
    return paths, banned


def _solve(k, budget):
    """Find k branches with disjoint interior paths, or None."""
    if k <= 2:
        return None
    best = None
    for combo in combinations(_ORDER, k):
        if _t.time() > budget:
            break
        combo = list(combo)
        banned = set(combo)
        # try to route; if greedy fails, backtrack by reordering branches
        paths, _ = _try_complete(combo, banned, budget)
        if paths is not None:
            return combo, paths
        # fallback: try a couple of alternative orderings
        for perm in [list(reversed(combo)), sorted(combo)]:
            if _t.time() > budget:
                break
            banned2 = set(perm)
            paths2, _ = _try_complete(perm, banned2, budget)
            if paths2 is not None:
                return perm, paths2
    return None


def construct():
    global _DEADLINE
    _DEADLINE = _t.time() + 80.0
    budget = _DEADLINE - 3.0
    # search from large k downward
    for k in range(_N, 2, -1):
        if _t.time() > budget:
            break
        res = _solve(k, budget)
        if res is not None:
            branches, paths = res
            out_paths = []
            for i in range(len(branches)):
                for j in range(i + 1, len(branches)):
                    a = branches[i]; b = branches[j]
                    p = paths.get((a, b)) or paths.get((b, a))
                    out_paths.append(list(p))
            return {'branches': [int(x) for x in branches],
                    'paths': [list(int(v) for v in pp) for pp in out_paths]}
    # ultimate fallback: an edge
    u, v = _EDGES[0]
    return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
# EVOLVE-BLOCK-END
