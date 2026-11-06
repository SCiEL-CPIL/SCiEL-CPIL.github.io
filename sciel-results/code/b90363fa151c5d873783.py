"""Hamilton cycle appearing early in an edge process.

Deterministic binary search over the edge-arrival threshold plus a
randomized DFS that extracts a Hamilton cycle from the induced prefix graph.
"""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGE_ORDER = P['edge_order']
M = len(EDGE_ORDER)

# Precompute a fast edge -> arrival-index lookup (1-based).
_EDGE_IDX = {}
for _i, (_a, _b) in enumerate(EDGE_ORDER, 1):
    if _a > _b:
        _a, _b = _b, _a
    _EDGE_IDX[(_a, _b)] = _i

# Precompute the 2-core of every prefix for the binary-search predicate.
_CORES = [None] * (M + 1)
for _k in range(1, M + 1):
    _adj = [set() for _ in range(N)]
    for _i in range(_k):
        _a, _b = EDGE_ORDER[_i]
        _adj[_a].add(_b)
        _adj[_b].add(_a)
    _deg = [len(_s) for _s in _adj]
    _alive = [True] * N
    _queue = [v for v in range(N) if _deg[v] < 2]
    while _queue:
        v = _queue.pop()
        if not _alive[v]:
            continue
        _alive[v] = False
        for w in _adj[v]:
            if _alive[w]:
                _deg[w] -= 1
                if _deg[w] < 2:
                    _queue.append(w)
    _CORES[_k] = _alive


def _cycle_score(cycle):
    """Largest one-based arrival index among the cycle's edges."""
    n = len(cycle)
    best = 0
    for i in range(n):
        a = cycle[i]
        b = cycle[(i + 1) % n]
        if a > b:
            a, b = b, a
        idx = _EDGE_IDX[(a, b)]
        if idx > best:
            best = idx
    return best


def _dfs_cycle(adj, deadline):
    """Randomized DFS for a Hamilton cycle in the induced prefix graph."""
    rng = random.Random(614)
    starts = list(range(N))
    for _ in range(20000):
        if time.time() > deadline:
            return None
        start = rng.choice(starts)
        parent = [-1] * N
        visited = [False] * N
        visited[start] = True
        stack = [start]
        while stack:
            if time.time() > deadline:
                return None
            v = stack[-1]
            nbrs = [w for w in adj[v] if not visited[w]]
            if nbrs:
                rng.shuffle(nbrs)
                w = nbrs[0]
                visited[w] = True
                parent[w] = v
                stack.append(w)
            else:
                stack.pop()
                if len(stack) == N:
                    cycle = list(stack)
                    end = cycle[-1]
                    st = cycle[0]
                    if st in adj[end]:
                        return cycle
    return None


def _find_cycle(k, deadline):
    adj = [set() for _ in range(N)]
    for i in range(k):
        a, b = EDGE_ORDER[i]
        adj[a].add(b)
        adj[b].add(a)
    return _dfs_cycle(adj, deadline)


def construct():
    deadline = time.time() + 75.0  # safety margin vs the 90s kill

    # Binary search the smallest k whose 2-core is the full vertex set.
    lo, hi = 2, M
    while lo < hi:
        mid = (lo + hi) // 2
        if _CORES[mid] == [True] * N:
            hi = mid
        else:
            lo = mid + 1
    k = lo

    # Try to extract a Hamilton cycle from the prefix graph at k.
    best_cycle = _find_cycle(k, deadline)
    if best_cycle is not None:
        return list(best_cycle)

    # Fallback: walk upward in k until a cycle is found (or time runs out).
    for kk in range(k + 1, M + 1):
        if time.time() > deadline:
            break
        cyc = _find_cycle(kk, deadline)
        if cyc is not None:
            return list(cyc)

    # Last-resort valid output: the identity Hamilton cycle.
    return list(range(N))
# EVOLVE-BLOCK-END
