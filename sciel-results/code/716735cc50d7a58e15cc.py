"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

N = P['n']
EDGES = P['edges']
ADJ = [set() for _ in range(N)]
for a, b in EDGES:
    ADJ[a].add(b)
    ADJ[b].add(a)

_T0 = time.time()
_DEADLINE = _T0 + 80.0   # hard wall-clock guard (kill is ~90s)


def _now():
    return time.time()


def _bfs_path(src, dst, forbidden):
    """Shortest path src->dst avoiding forbidden interior vertices. Returns list or None."""
    if src == dst:
        return [src]
    if src in forbidden:
        return None
    parent = {src: None}
    q = deque([src])
    while q:
        u = q.popleft()
        if u == dst:
            break
        for v in ADJ[u]:
            if v in parent:
                continue
            if v != dst and v in forbidden:
                continue
            parent[v] = u
            q.append(v)
    if dst not in parent:
        return None
    path = []
    cur = dst
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def _bron_kerbosch():
    """Return a maximum clique (list of vertex ids)."""
    adj = [set() for _ in range(N)]
    for a, b in EDGES:
        adj[a].add(b)
        adj[b].add(a)
    best = []
    def bk(R, Pset, X):
        nonlocal best
        if not Pset and not X:
            if len(R) > len(best):
                best = list(R)
            return
        if len(R) + len(Pset) <= len(best):
            return
        # pivot
        u = max(Pset | X, key=lambda v: len(adj[v] & Pset)) if (Pset | X) else None
        candidates = list(Pset - adj[u]) if u is not None else list(Pset)
        for v in candidates:
            if _now() > _DEADLINE:
                return
            R.add(v)
            bk(R, Pset & adj[v], X & adj[v])
            R.remove(v)
            Pset.remove(v)
            X.add(v)
    bk(set(), set(range(N)), set())
    return best


def _can_add(cand, branches, forbidden):
    """Try to add cand to branches; returns list of new paths or None if impossible."""
    new_paths = []
    # order existing branches by degree (harder to route to first)
    ordered = sorted(branches, key=lambda b: len(ADJ[b]))
    for b in ordered:
        p = _bfs_path(cand, b, forbidden)
        if p is None:
            return None
        new_paths.append(p)
        forbidden = forbidden | set(p[1:-1])
    return new_paths


def _try_seed(seed):
    """Given an initial branch set, greedily extend. Returns (branches, paths) or None."""
    branches = list(seed)
    forbidden = set(branches)
    paths = []
    # direct edges among seed
    for a, b in combinations(branches, 2):
        if b in ADJ[a]:
            paths.append([a, b])
        else:
            return None
    # greedy extension
    remaining = [v for v in range(N) if v not in forbidden]
    while remaining:
        if _now() > _DEADLINE:
            break
        best_cand = None
        best_paths = None
        # sort candidates: prefer low-degree (easier to route) first
        for cand in sorted(remaining, key=lambda v: len(ADJ[v])):
            np_ = _can_add(cand, branches, forbidden)
            if np_ is not None:
                best_cand = cand
                best_paths = np_
                break
        if best_cand is None:
            break
        branches.append(best_cand)
        forbidden.add(best_cand)
        paths.extend(best_paths)
        remaining.remove(best_cand)
    return (branches, paths)


def _try_seed_order(seed, order):
    """Build branches by adding in given order; return (branches, paths) or None."""
    branches = []
    forbidden = set()
    paths = []
    for v in order:
        if v in forbidden:
            continue
        if not branches:
            branches.append(v)
            forbidden.add(v)
            continue
        np_ = _can_add(v, branches, forbidden)
        if np_ is None:
            continue
        branches.append(v)
        forbidden.add(v)
        paths.extend(np_)
    if len(branches) < 2:
        return None
    return (branches, paths)


def _validate(branches, paths):
    if branches is None:
        return False
    if len(branches) < 2:
        return False
    bset = set(branches)
    if len(bset) != len(branches):
        return False
    # every unordered pair must have exactly one path
    pairs = {}
    for p in paths:
        if not isinstance(p, list):
            return False
        if len(p) < 2:
            return False
        a, b = p[0], p[-1]
        if a not in bset or b not in bset:
            return False
        if a == b:
            return False
        key = (min(a, b), max(a, b))
        if key in pairs:
            return False
        pairs[key] = p
    # all pairs covered
    for a, b in combinations(sorted(bset), 2):
        if (a, b) not in pairs:
            return False
    # interiors disjoint and avoid branches
    used = set()
    for key, p in pairs.items():
        for v in p[1:-1]:
            if v in bset:
                return False
            if v in used:
                return False
            used.add(v)
    # adjacency
    for key, p in pairs.items():
        for i in range(len(p) - 1):
            if p[i + 1] not in ADJ[p[i]]:
                return False
    return True


def construct():
    global _DEADLINE
    _T0 = time.time()
    _DEADLINE = _T0 + 80.0

    best = None
    best_size = -1

    def consider(branches, paths):
        nonlocal best, best_size
        if branches is None:
            return
        if len(branches) <= best_size:
            return
        if not _validate(branches, paths):
            return
        best = (branches, paths)
        best_size = len(branches)

    # 1) maximum clique seed
    clique = _bron_kerbosch()
    if clique:
        res = _try_seed(clique)
        if res:
            consider(*res)

    # 2) random orderings (structural: treat as vertex-disjoint routing problem)
    rng = random.Random(12345)
    all_v = list(range(N))
    attempts = 0
    max_attempts = 400
    while attempts < max_attempts and _now() < _DEADLINE:
        order = all_v[:]
        rng.shuffle(order)
        res = _try_seed_order([], order)
        if res:
            consider(*res)
        attempts += 1

    # 3) degree-ordered ordering (deterministic)
    deg_order = sorted(range(N), key=lambda v: len(ADJ[v]))
    res = _try_seed_order([], deg_order)
    if res:
        consider(*res)
    res = _try_seed_order([], list(reversed(deg_order)))
    if res:
        consider(*res)

    # fallback: guarantee a valid 2-branch result
    if best is None:
        a, b = EDGES[0]
        best = ([a, b], [[a, b]])

    branches, paths = best
    return {
        'branches': [int(x) for x in branches],
        'paths': [[int(x) for x in p] for p in paths],
    }
# EVOLVE-BLOCK-END
