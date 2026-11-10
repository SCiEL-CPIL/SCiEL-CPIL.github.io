"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import sys
from collections import deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

N = P['n']
EDGE_SET = {tuple(e) for e in P['edges']}
ADJ = [set() for _ in range(N)]
for u, v in P['edges']:
    ADJ[u].add(v)
    ADJ[v].add(u)
DEADLINE = 55.0


def _is_edge(u, v):
    return (u, v) in EDGE_SET or (v, u) in EDGE_SET


def _maxflow(s, t, forbidden, cap):
    """Max flow s->t in unit graph; interior vertices forbidden, capacity cap."""
    V = N + 2
    SRC = N
    SNK = N + 1
    adj = [[] for _ in range(V)]
    capd = [[0] * V for _ in range(V)]
    rev = {}

    def add_edge(a, b, c):
        adj[a].append(b)
        capd[a][b] += c
        adj[b].append(a)
        rev[(b, a)] = rev.get((a, b), 0) + 1

    for u in range(N):
        if u == s or u == t or u in forbidden:
            continue
        add_edge(u, u + 1, 1)
    for u, v in P['edges']:
        if u == s or u == t or v == s or v == t:
            continue
        if u in forbidden or v in forbidden:
            continue
        add_edge(u + 1, v, cap)
        add_edge(v + 1, u, cap)
    add_edge(SRC, s + 1, cap)
    add_edge(t, SNK, cap)

    def bfs():
        dist = [-1] * V
        dist[SRC] = 0
        q = deque([SRC])
        while q:
            x = q.popleft()
            for y in adj[x]:
                if dist[y] < 0 and capd[x][y] > 0:
                    dist[y] = dist[x] + 1
                    q.append(y)
        return dist

    flow = 0
    while True:
        dist = bfs()
        if dist[SNK] < 0:
            break
        it = [0] * V

        def dfs(x, f):
            if x == SNK:
                return f
            while it[x] < len(adj[x]):
                y = adj[x][it[x]]
                if capd[x][y] > 0 and dist[x] + 1 == dist[y]:
                    ret = dfs(y, min(f, capd[x][y]))
                    if ret > 0:
                        capd[x][y] -= ret
                        capd[y][x] += ret
                        return ret
                it[x] += 1
            return 0

        while True:
            pushed = dfs(SRC, 10 ** 9)
            if pushed == 0:
                break
            flow += pushed
    return flow


def _bfs_path(s, t, forbidden):
    if s == t:
        return None
    if _is_edge(s, t):
        return [s, t]
    prev = {s: None}
    q = deque([s])
    while q:
        u = q.popleft()
        if u == t:
            break
        for v in sorted(ADJ[u]):
            if v in prev:
                continue
            if v in forbidden and v != t:
                continue
            prev[v] = u
            q.append(v)
    if t not in prev:
        return None
    path = []
    cur = t
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _try_paths(branches, order):
    """Attempt to find internally-disjoint paths for all pairs, in given order.
    Returns dict pair->path or None."""
    forbidden = set(branches)
    paths = {}
    for (a, b) in order:
        p = _bfs_path(a, b, forbidden)
        if p is None:
            return None
        paths[(a, b)] = p
        for x in p[1:-1]:
            forbidden.add(x)
    return paths


def _try_paths_flow(branches, order):
    """Max-flow based: verify via flow, extract paths via successive BFS."""
    forbidden = set(branches)
    paths = {}
    for (a, b) in order:
        p = _bfs_path(a, b, forbidden)
        if p is None:
            return None
        paths[(a, b)] = p
        for x in p[1:-1]:
            forbidden.add(x)
    return paths


def _greedy_expand(branches, order, forbidden, paths):
    """Try to add candidate vertices one at a time, rerouting as needed."""
    branches = list(branches)
    order = list(order)
    forbidden = set(forbidden)
    paths = dict(paths)
    changed = True
    while changed:
        changed = False
        for cand in range(N):
            if cand in branches:
                continue
            nb = ADJ[cand]
            # candidate must connect to all existing branches
            if not all(b in nb for b in branches):
                continue
            # try adding candidate
            new_branches = branches + [cand]
            new_paths = dict(paths)
            new_order = list(order)
            # add new pairs involving candidate
            for b in branches:
                key = (min(cand, b), max(cand, b))
                new_order.append((cand, b))
                new_paths[key] = [cand, b]  # direct edge
            # need to verify all disjoint: rebuild with BFS in new order
            forb2 = set(new_branches)
            ok = True
            for (a, b) in new_order:
                p = _bfs_path(a, b, forb2)
                if p is None:
                    ok = False
                    break
                new_paths[(a, b)] = p
                for x in p[1:-1]:
                    forb2.add(x)
            if ok:
                branches = new_branches
                order = new_order
                forbidden = forb2
                paths = new_paths
                changed = True
                break
    return branches, order, forbidden, paths


def _random_order(branches):
    pairs = list(combinations(branches, 2))
    random.shuffle(pairs)
    return pairs


def _build_from_branches(branches, order):
    forbidden = set(branches)
    paths = {}
    for (a, b) in order:
        p = _bfs_path(a, b, forbidden)
        if p is None:
            return None
        paths[(a, b)] = p
        for x in p[1:-1]:
            forbidden.add(x)
    return branches, order, forbidden, paths


def _score(branches, order, forbidden, paths):
    return len(branches)


def _make_result(branches, order, paths):
    paths_out = []
    for (a, b) in order:
        paths_out.append(list(paths[(a, b)]))
    return {'branches': list(branches), 'paths': paths_out}


def construct():
    t0 = time.time()
    rng = random.Random(12345)

    best = None
    best_size = 1

    def consider(branches, order, forbidden, paths):
        nonlocal best, best_size
        if len(branches) > best_size:
            best_size = len(branches)
            best = _make_result(branches, order, paths)

    # --- Strategy 1: random branch set + expansion ---
    for trial in range(400):
        if time.time() > DEADLINE:
            break
        # pick random starting clique-ish set
        k = rng.randint(2, 4)
        start = rng.sample(range(N), k)
        # verify pairwise edges
        if not all(_is_edge(a, b) for a, b in combinations(start, 2)):
            continue
        order = _random_order(start)
        res = _build_from_branches(start, order)
        if res is None:
            continue
        branches, order, forbidden, paths = res
        branches, order, forbidden, paths = _greedy_expand(branches, order, forbidden, paths)
        consider(branches, order, forbidden, paths)

    # --- Strategy 2: start from random high-degree vertex pairs, expand ---
    deg = [len(ADJ[u]) for u in range(N)]
    ranked = sorted(range(N), key=lambda u: -deg[u])
    for trial in range(300):
        if time.time() > DEADLINE:
            break
        # pick a random base vertex among top-degree
        base = ranked[rng.randint(0, min(10, N - 1))]
        # build a random neighbor set that is pairwise connected
        nbrs = list(ADJ[base])
        rng.shuffle(nbrs)
        start = [base]
        for nb in nbrs:
            if all(_is_edge(nb, x) for x in start):
                start.append(nb)
            if len(start) >= 4:
                break
        if len(start) < 2:
            continue
        order = _random_order(start)
        res = _build_from_branches(start, order)
        if res is None:
            continue
        branches, order, forbidden, paths = res
        branches, order, forbidden, paths = _greedy_expand(branches, order, forbidden, paths)
        consider(branches, order, forbidden, paths)

    # --- Strategy 3: random 2-vertex start, expand to a big set, then try to add ---
    for trial in range(300):
        if time.time() > DEADLINE:
            break
        a = rng.randint(0, N - 1)
        b = rng.choice(list(ADJ[a]))
        start = [a, b]
        order = _random_order(start)
        res = _build_from_branches(start, order)
        if res is None:
            continue
        branches, order, forbidden, paths = res
        branches, order, forbidden, paths = _greedy_expand(branches, order, forbidden, paths)
        consider(branches, order, forbidden, paths)

    # --- Strategy 4: random seed of 3 random vertices that form a path/triangle ---
    for trial in range(200):
        if time.time() > DEADLINE:
            break
        a = rng.randint(0, N - 1)
        b = rng.choice(list(ADJ[a]))
        cands = [c for c in ADJ[b] if c != a and _is_edge(c, a)]
        if not cands:
            continue
        c = rng.choice(cands)
        start = [a, b, c]
        order = _random_order(start)
        res = _build_from_branches(start, order)
        if res is None:
            continue
        branches, order, forbidden, paths = res
        branches, order, forbidden, paths = _greedy_expand(branches, order, forbidden, paths)
        consider(branches, order, forbidden, paths)

    # --- Strategy 5: local search / reordering to unstick ---
    if best is not None:
        # try to improve best by reordering
        for trial in range(150):
            if time.time() > DEADLINE:
                break
            branches = best['branches']
            # random order
            order = _random_order(branches)
            res = _build_from_branches(branches, order)
            if res is not None:
                branches, order, forbidden, paths = res
                branches, order, forbidden, paths = _greedy_expand(branches, order, forbidden, paths)
                consider(branches, order, forbidden, paths)

    # --- Fallback: find any triangle or edge ---
    if best is None:
        for row in combinations(range(N), 3):
            if all(_is_edge(a, b) for a, b in combinations(row, 2)):
                order = list(combinations(row, 2))
                res = _build_from_branches(list(row), order)
                if res is not None:
                    best = _make_result(*res[:3])
                    break
    if best is None:
        u, v = P['edges'][0]
        best = {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}

    # Ensure types are plain ints/lists
    best['branches'] = [int(x) for x in best['branches']]
    best['paths'] = [[int(x) for x in p] for p in best['paths']]
    return best
# EVOLVE-BLOCK-END
