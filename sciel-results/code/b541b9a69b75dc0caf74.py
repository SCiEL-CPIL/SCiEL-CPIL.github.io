"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
from collections import deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

_DEADLINE = _time.monotonic() + 78.0


def _time_left():
    return _time.monotonic() < _DEADLINE


def construct():
    n = P['n']
    edges = P['edges']

    adj = [set() for _ in range(n)]
    for a, b in edges:
        a = int(a)
        b = int(b)
        adj[a].add(b)
        adj[b].add(a)

    # ---- guaranteed 2-branch fallback (direct edge) ----
    u0, v0 = int(edges[0][0]), int(edges[0][1])
    best_branches = [u0, v0]
    best_paths = [[u0, v0]]
    best_k = 2

    def is_clique(S):
        for x, y in combinations(S, 2):
            if y not in adj[x]:
                return False
        return True

    def try_realize_paths(branches, max_len):
        """Try to find one simple path per unordered branch pair.
        Path endpoints are the pair; interiors must be pairwise disjoint
        and avoid all branches. Return list of paths or None."""
        Sset = set(branches)
        pairs = list(combinations(branches, 2))
        m = len(pairs)

        # Greedy ordering of pairs: try pairs whose common non-branch neighbors are fewest first?
        # We'll just do pairs in order; BFS with dynamic forbidden set.
        used_interior = set()
        paths = []
        for (a, b) in pairs:
            # BFS from a to b avoiding Sset and used_interior in the interior
            # (endpoints a,b are allowed as they are branches)
            parent = {}
            dist = {}
            q = deque([a])
            dist[a] = 0
            parent[a] = None
            found = False
            while q:
                cur = q.popleft()
                if cur == b:
                    found = True
                    break
                # don't expand past max_len-1 edges (path length in edges)
                if dist[cur] >= max_len - 1:
                    continue
                for nb in adj[cur]:
                    if nb == a or nb == b:
                        # only allow reaching b as target; a is start
                        if nb == b and cur != a:
                            pass  # allowed
                        elif nb == a:
                            continue
                        else:
                            continue
                    # interior constraint: nb not in Sset and not used_interior
                    if nb in Sset:
                        # b is allowed as target only; other branches not allowed as interior
                        if nb == b:
                            pass
                        else:
                            continue
                    if nb in used_interior:
                        continue
                    if nb not in dist:
                        dist[nb] = dist[cur] + 1
                        parent[nb] = cur
                        q.append(nb)
            if not found:
                return None
            # reconstruct path
            path = []
            cur = b
            while cur is not None:
                path.append(cur)
                if cur == a:
                    break
                cur = parent[cur]
            path.reverse()
            # sanity: length in edges = len(path)-1 <= max_len
            if len(path) - 1 > max_len:
                return None
            # mark interiors
            for v in path[1:-1]:
                if v in Sset or v in used_interior:
                    return None
                used_interior.add(v)
            paths.append(path)
        return paths

    def try_realize_paths_len2(branches):
        """Specialized fast path for max_len=2 (single Steiner per pair)."""
        Sset = set(branches)
        pairs = list(combinations(branches, 2))
        m = len(pairs)
        steiners = [v for v in range(n) if v not in Sset]
        if len(steiners) < m:
            return None
        # build adjacency for matching
        idx = {s: i for i, s in enumerate(steiners)}
        adjL = [[] for _ in range(m)]
        for i, (a, b) in enumerate(pairs):
            row = []
            for s in steiners:
                if s in adj[a] and s in adj[b]:
                    row.append(idx[s])
            adjL[i] = row
        # Hopcroft-Karp
        INF = 1 << 60
        Lmatch = [-1] * m
        Rmatch = [-1] * len(steiners)
        dist = [0] * m
        found = 0

        def bfs():
            dq = deque()
            for u in range(m):
                if Lmatch[u] == -1:
                    dist[u] = 0
                    dq.append(u)
                else:
                    dist[u] = INF
            d_end = INF
            while dq:
                u = dq.popleft()
                if dist[u] < d_end:
                    for v in adjL[u]:
                        w = Rmatch[v]
                        if w == -1:
                            d_end = dist[u] + 1
                        elif dist[w] == INF:
                            dist[w] = dist[u] + 1
                            dq.append(w)
            return d_end != INF

        def dfs(u):
            for v in adjL[u]:
                w = Rmatch[v]
                if w == -1 or (dist[w] == dist[u] + 1 and dfs(w)):
                    Lmatch[u] = v
                    Rmatch[v] = u
                    return True
            dist[u] = INF
            return False

        while bfs():
            for u in range(m):
                if Lmatch[u] == -1 and dfs(u):
                    found += 1
        if found != m:
            return None
        paths = []
        for i, (a, b) in enumerate(pairs):
            s = steiners[Lmatch[i]]
            paths.append([int(a), int(s), int(b)])
        return paths

    # ---- greedy incremental branch growth from many seeds ----
    # Candidate order: by descending degree, with some randomization via multiple passes
    order_by_deg = sorted(range(n), key=lambda v: len(adj[v]), reverse=True)

    def greedy_from_seed(seed_list):
        """Grow a clique starting from seed_list, trying to add vertices,
        then realize paths. Return (branches, paths) or None."""
        cur = list(seed_list)
        cur_set = set(cur)
        if not is_clique(cur):
            return None
        # try adding vertices in order_by_deg
        for v in order_by_deg:
            if v in cur_set:
                continue
            if not _time_left():
                break
            # v must be adjacent to all current
            ok = True
            for x in cur:
                if v not in adj[x]:
                    ok = False
                    break
            if not ok:
                continue
            trial = cur + [v]
            # try to realize paths with len 2 first (fast), then longer
            res = try_realize_paths_len2(trial)
            if res is None:
                # try longer paths, but bounded to keep runtime
                res = try_realize_paths(trial, max_len=4)
            if res is not None:
                cur = trial
                cur_set.add(v)
            else:
                # don't add
                continue
        if len(cur) >= 2:
            res = try_realize_paths_len2(cur)
            if res is None:
                res = try_realize_paths(cur, max_len=4)
            if res is not None:
                return cur, res
        return None

    # Seeds: single high-degree vertex, pairs, triples
    seeds = []
    for v in order_by_deg[:12]:
        seeds.append([v])
    for a, b in combinations(order_by_deg[:10], 2):
        if b in adj[a]:
            seeds.append([a, b])
    for a, b, c in combinations(order_by_deg[:8], 3):
        if is_clique([a, b, c]):
            seeds.append([a, b, c])

    for seed in seeds:
        if not _time_left():
            break
        got = greedy_from_seed(seed)
        if got is None:
            continue
        branches, paths = got
        k = len(branches)
        if k > best_k:
            best_k = k
            best_branches = [int(x) for x in branches]
            best_paths = [[int(x) for x in p] for p in paths]
        # early exit if we reached a very large set
        if best_k >= 10:
            break

    # ---- also try the original descending-size enumeration but with len-2 matching only (fast) ----
    if _time_left():
        verts = list(range(n))
        for k in range(best_k + 1, n, 1):
            if not _time_left():
                break
            found_any = False
            for comb in combinations(verts, k):
                if not _time_left():
                    break
                if not is_clique(comb):
                    continue
                res = try_realize_paths_len2(list(comb))
                if res is not None:
                    best_k = k
                    best_branches = [int(x) for x in comb]
                    best_paths = [[int(x) for x in p] for p in res]
                    found_any = True
                    break
            if found_any:
                break

    # final type conversion
    best_branches = [int(x) for x in best_branches]
    best_paths = [[int(x) for x in p] for p in best_paths]

    if len(best_branches) < 2:
        best_branches = [u0, v0]
        best_paths = [[u0, v0]]

    return {'branches': best_branches, 'paths': best_paths}
# EVOLVE-BLOCK-END
