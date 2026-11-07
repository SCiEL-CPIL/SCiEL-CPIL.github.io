import time
import random
import itertools
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']

    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    start = time.time()
    deadline = start + 82.0  # safety margin under 90s

    def time_left():
        return deadline - time.time()

    # ---------- Max clique via Bron-Kerbosch with pivot ----------
    def find_max_clique():
        best = []

        def bronk(R, Pset, X):
            if time.time() > deadline:
                return
            if not Pset and not X:
                if len(R) > len(best):
                    best[:] = list(R)
                return
            if Pset or X:
                pivot = max(Pset | X, key=lambda v: len(adj[v] & Pset))
            else:
                return
            candidates = [v for v in Pset if v not in adj[pivot]]
            for v in candidates:
                bronk(R | {v}, Pset & adj[v], X & adj[v])
                Pset = Pset - {v}
                X = X | {v}

        bronk(frozenset(), frozenset(range(n)), frozenset())
        return best

    # ---------- Candidate path generation ----------
    def gen_paths(u, v, branch_set, max_len, cap):
        """Simple paths from u to v, avoiding branch_set in the interior, length <= max_len."""
        out = []
        if u == v:
            return out

        def dfs(path, visited):
            if len(out) >= cap:
                return
            if time.time() > deadline:
                return
            cur = path[-1]
            if cur == v:
                if len(path) - 1 <= max_len:
                    out.append(list(path))
                return
            if len(path) - 1 >= max_len:
                return
            for nb in sorted(adj[cur]):
                if nb == v:
                    if len(path) <= max_len:
                        out.append(path + [nb])
                    continue
                if nb not in visited and nb not in branch_set:
                    visited.add(nb)
                    dfs(path + [nb], visited)
                    visited.remove(nb)

        dfs([u], {u})
        out.sort(key=lambda x: (len(x), x))
        return out

    def route(branches, max_len, cap):
        """Backtrack assignment of internally-disjoint paths for all pairs of branches.
        Returns dict {frozenset pair: path} or None."""
        branch_set = set(branches)
        pairs = list(itertools.combinations(branches, 2))
        cands = {}
        for u, v in pairs:
            cands[(u, v)] = gen_paths(u, v, branch_set, max_len, cap)
        for u, v in pairs:
            if not cands[(u, v)]:
                return None

        order = sorted(pairs, key=lambda p: len(cands[p]))
        used = set()
        assign = {}

        def bt(i):
            if time.time() > deadline:
                return False
            if i == len(order):
                return True
            u, v = order[i]
            for path in cands[(u, v)]:
                interior = set(path[1:-1])
                if interior & used:
                    continue
                assign[(u, v)] = path
                used.update(interior)
                if bt(i + 1):
                    return True
                used.difference_update(interior)
                del assign[(u, v)]
            return False

        if bt(0):
            return {frozenset(p): assign[p] for p in pairs}
        return None

    # ---------- Verification ----------
    def valid(branches, paths):
        if len(branches) < 2:
            return False
        used = set()
        bset = set(branches)
        for b1, b2 in itertools.combinations(branches, 2):
            p = paths.get(frozenset((b1, b2)))
            if p is None:
                return False
            if p[0] not in (b1, b2) or p[-1] not in (b1, b2):
                return False
            interior = set(p[1:-1])
            if interior & used:
                return False
            if interior & bset:
                return False
            used.update(interior)
        return True

    best_branches = None
    best_paths = None
    best_size = 0

    def record(branches, paths):
        nonlocal best_branches, best_paths, best_size
        if len(branches) > best_size and valid(branches, paths):
            best_size = len(branches)
            best_branches = list(branches)
            best_paths = {tuple(sorted(k)): list(v) for k, v in paths.items()}

    # ---------- Phase 1: max clique as a trivial subdivision ----------
    clique = find_max_clique()
    if len(clique) >= 2:
        cl = sorted(clique)
        paths = {frozenset((u, v)): [u, v] for u, v in itertools.combinations(cl, 2)}
        record(cl, paths)

    # ---------- Phase 2: expand clique by adding vertices, rerouting with backtracking ----------
    degrees = {v: len(adj[v]) for v in range(n)}
    for seed in range(8):
        if time_left() < 3.0:
            break
        base = list(clique)
        if seed > 0:
            random.Random(seed).shuffle(base)
        # Try to grow by adding vertices in degree order, rerouting each time
        cur = list(base)
        for v in sorted(range(n), key=lambda x: -degrees[x]):
            if time_left() < 3.0:
                break
            if v in cur:
                continue
            cand = cur + [v]
            # try a few max_len values
            got = None
            for ml in (3, 4, 5):
                if time_left() < 2.0:
                    break
                got = route(cand, ml, cap=12)
                if got is not None:
                    break
            if got is not None:
                cur = cand
                record(cur, got)

    # ---------- Phase 3: randomized branch sets, direct backtracking route ----------
    rng = random.Random(12345)
    for it in range(4000):
        if time_left() < 2.0:
            break
        # random size between best_size-1 and 9
        lo = max(2, best_size - 1)
        hi = min(n, 9)
        if hi < lo:
            hi = lo
        k = rng.randint(lo, hi)
        branches = rng.sample(range(n), k)
        got = route(branches, max_len=4, cap=8)
        if got is not None:
            record(branches, got)

    # ---------- Fallback ----------
    if best_branches is None:
        best_branches = [0, 1]
        best_paths = {(0, 1): [0, 1]}

    output_paths = []
    for b1, b2 in itertools.combinations(best_branches, 2):
        key = (min(b1, b2), max(b1, b2))
        p = best_paths.get(key)
        if p is None:
            p = [b1, b2]
        output_paths.append(list(p))

    return {
        'branches': [int(x) for x in best_branches],
        'paths': output_paths,
    }
