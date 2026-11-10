"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import random as _random
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}


def construct():
    start_time = _time.time()
    deadline = start_time + 85  # safety margin before the 90s kill
    n = P['n']
    edges = P['edges']

    adj = [set() for _ in range(n)]
    adj_list = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        adj_list[u].append(v)
        adj_list[v].append(u)

    def time_check():
        return _time.time() > deadline

    # ---------------- max-flow (Dinic) ----------------
    class _Dinic:
        def __init__(self, size):
            self.size = size
            self.cap = [[0] * size for _ in range(size)]
            self.lvl = [0] * size
            self.it = [0] * size

        def add_edge(self, u, v, c):
            self.cap[u][v] += c
            self.cap[v][u] += 0

        def _bfs(self, s, t):
            self.lvl = [-1] * self.size
            self.lvl[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                for v in range(self.size):
                    if self.lvl[v] < 0 and self.cap[u][v] > 0:
                        self.lvl[v] = self.lvl[u] + 1
                        q.append(v)
            return self.lvl[t] >= 0

        def _dfs(self, u, t, f):
            if u == t:
                return f
            for _ in range(self.it[u], self.size):
                v = self.it[u]
                if self.lvl[v] == self.lvl[u] + 1 and self.cap[u][v] > 0:
                    d = self._dfs(v, t, f if f < self.cap[u][v] else self.cap[u][v])
                    if d > 0:
                        self.cap[u][v] -= d
                        self.cap[v][u] += d
                        return d
                self.it[u] += 1
            return 0

        def max_flow(self, s, t):
            flow = 0
            INF = 10 ** 9
            while self._bfs(s, t):
                self.it = [0] * self.size
                while True:
                    f = self._dfs(s, t, INF)
                    if f == 0:
                        break
                    flow += f
            return flow

    # ---------------- length-2 (bipartite matching) path check ----------------
    # Returns a dict {(u,v): path} for every unordered pair, or None if infeasible.
    # Uses Hopcroft-Karp to assign a distinct common-neighbor to each non-adjacent pair.
    def check_len2(branches):
        bset = set(branches)
        pairs = list(combinations(sorted(branches), 2))
        non_direct = []
        for u, v in pairs:
            if u in adj[v]:
                continue
            common = [w for w in adj_list[u] if w not in bset and w in adj[v]]
            if not common:
                return None
            non_direct.append((u, v, common))
        if not non_direct:
            return {(u, v): [u, v] for u, v in pairs}

        # Build bipartite graph: left = non_direct pair index, right = common-neighbor vertex
        # Hopcroft-Karp
        left_n = len(non_direct)
        right_ids = {}
        for idx, (u, v, common) in enumerate(non_direct):
            for w in common:
                if w not in right_ids:
                    right_ids[w] = len(right_ids)
        right_n = len(right_ids)

        L_adj = [[] for _ in range(left_n)]
        for idx, (u, v, common) in enumerate(non_direct):
            for w in common:
                L_adj[idx].append(right_ids[w])

        pairU = [-1] * left_n
        pairV = [-1] * right_n
        dist = [0] * left_n

        def _bfs_hk():
            q = deque()
            for u in range(left_n):
                if pairU[u] == -1:
                    dist[u] = 0
                    q.append(u)
                else:
                    dist[u] = -1
            found = False
            while q:
                u = q.popleft()
                for v in L_adj[u]:
                    pu = pairV[v]
                    if pu == -1:
                        found = True
                    elif dist[pu] == -1:
                        dist[pu] = dist[u] + 1
                        q.append(pu)
            return found

        def _dfs_hk(u):
            for v in L_adj[u]:
                pu = pairV[v]
                if pu == -1 or (dist[pu] == dist[u] + 1 and _dfs_hk(pu)):
                    pairU[u] = v
                    pairV[v] = u
                    return True
            dist[u] = -1
            return False

        while _bfs_hk():
            for u in range(left_n):
                if pairU[u] == -1:
                    _dfs_hk(u)
                    if time_check():
                        return None

        if any(pairU[u] == -1 for u in range(left_n)):
            return None

        result = {}
        for u, v in pairs:
            result[(u, v)] = [u, v]
        for idx, (u, v, common) in enumerate(non_direct):
            w = common[pairU[idx]]
            result[(u, v)] = [u, w, v]
        return result

    # ---------------- max-flow check for longer paths (feasibility + routing) ----------------
    def check_flow(branches):
        """Vertex-disjoint paths for all pairs via a flow network. Returns paths dict or None."""
        bset = set(branches)
        pairs = list(combinations(sorted(branches), 2))
        # Build split graph: each vertex v -> v_in, v_out; edge v_in->v_out cap 1 (interior),
        # cap len(branches) for branch vertices (they are endpoints of many paths).
        # For each graph edge (u,v): u_out -> v_in cap 1, v_out -> u_in cap 1.
        # Super source S -> branch_in, super sink branch_out -> T, capacity 1 per pair.
        # To route k pairs simultaneously, connect S->each branch_in with cap = (deg in pairs),
        # and branch_out -> T similarly; but we need exactly one unit per pair. Simplest:
        # one unit of flow per pair. Use S->u_in (cap 1) for each pair endpoint occurrence.
        # Instead: for each pair (u,v), add a dedicated source node? Simpler: aggregate.
        # We add S->u_in cap = (#pairs containing u) and u_out->T cap = (#pairs containing u),
        # then max flow should equal number of pairs.
        id_of = {}
        def node(v, side):  # side 0=in,1=out
            key = (v, side)
            if key not in id_of:
                id_of[key] = len(id_of)
            return id_of[key]

        S = len(id_of)
        T = len(id_of) + 1
        size = T + 1
        dinic = _Dinic(size)
        for v in range(n):
            cap = len(branches) if v in bset else 1
            dinic.add_edge(node(v, 0), node(v, 1), cap)
        for u, v in edges:
            dinic.add_edge(node(u, 1), node(v, 0), 1)
            dinic.add_edge(node(v, 1), node(u, 0), 1)
        cnt = {}
        for u, v in pairs:
            cnt[u] = cnt.get(u, 0) + 1
            cnt[v] = cnt.get(v, 0) + 1
        for v in bset:
            c = cnt.get(v, 0)
            if c:
                dinic.add_edge(S, node(v, 0), c)
                dinic.add_edge(node(v, 1), T, c)
        if dinic.max_flow(S, T) != len(pairs):
            return None
        # Recover one path per pair by walking residual (each pair uses 1 unit).
        # For each pair, find an interior-disjoint path in the used edges.
        used_edges = set()
        for u in range(size):
            for v in range(size):
                if dinic.cap[u][v] < 0:
                    used_edges.add((u, v))
        # Reverse map node id -> (vertex, side)
        inv = {val: key for key, val in id_of.items()}
        # Reconstruct: for each pair (u,v), find a path from u_in to v_out through used edges.
        # Build used adjacency on node ids.
        used_adj = {i: [] for i in id_of.values()}
        for a, b in used_edges:
            used_adj.setdefault(a, []).append(b)
        result = {}
        for u, v in pairs:
            # BFS from node(u,0) to node(v,1) within used_adj, preferring interior vertices
            start = node(u, 0)
            goal = node(v, 1)
            par = {start: None}
            q = deque([start])
            found = False
            while q:
                cur = q.popleft()
                if cur == goal:
                    found = True
                    break
                for nb in used_adj.get(cur, []):
                    if nb not in par:
                        par[nb] = cur
                        q.append(nb)
            if not found:
                return None
            seq = []
            cur = goal
            while cur is not None:
                seq.append(cur)
                cur = par[cur]
            seq.reverse()
            verts = []
            for nid in seq:
                vv, side = inv[nid]
                if side == 1:
                    verts.append(vv)
            # dedupe consecutive (in/out of same vertex)
            path = []
            for x in verts:
                if not path or path[-1] != x:
                    path.append(x)
            if path[0] != u or path[-1] != v:
                return None
            result[(u, v)] = path
        return result

    def verify(branches):
        """Return a valid paths dict or None. Try length-2 matching first (fast), then flow."""
        r = check_len2(branches)
        if r is not None:
            return r
        if time_check():
            return None
        return check_flow(branches)

    # ---------------- max clique (Bron-Kerbosch with pivot) ----------------
    def max_clique():
        best = []
        def bk(R, Pset, X):
            nonlocal best
            if time_check():
                return
            if not Pset and not X:
                if len(R) > len(best):
                    best = list(R)
                return
            pivot_cands = list(Pset | X)
            if pivot_cands:
                pivot = max(pivot_cands, key=lambda v: len(adj[v] & Pset))
            else:
                return
            for v in [x for x in Pset if x not in adj[pivot]]:
                bk(R | {v}, Pset & adj[v], X & adj[v])
                Pset.discard(v)
                X.add(v)
        bk(set(), set(range(n)), set())
        return best

    # ---------------- growth & local search (refinement machinery) ----------------
    def try_grow(initial):
        current = list(initial)
        improved = True
        while improved and not time_check():
            improved = False
            candidates = [v for v in range(n) if v not in current]
            def score(v):
                return sum(1 for u in current if v in adj[u])
            candidates.sort(key=score, reverse=True)
            for cand in candidates:
                new_set = current + [cand]
                if verify(new_set) is not None:
                    current = new_set
                    improved = True
                    break
        return current

    def local_search(branches):
        current = list(branches)
        improved = True
        while improved and not time_check():
            improved = False
            for i in range(len(current)):
                if time_check():
                    break
                for cand in range(n):
                    if cand in current:
                        continue
                    new_set = current[:i] + [cand] + current[i+1:]
                    if verify(new_set) is not None:
                        current = new_set
                        improved = True
                        break
                if improved:
                    break
        return current

    # ---------------- orchestration ----------------
    best_branches = None
    best_paths = None

    def update_best(b, p):
        nonlocal best_branches, best_paths
        if b is None or p is None:
            return
        if best_branches is None or len(b) > len(best_branches):
            best_branches = list(b)
            best_paths = p

    # Strategy 1: maximum clique seed
    mc = max_clique()
    if len(mc) >= 2:
        grown = try_grow(mc)
        update_best(grown, verify(grown))

    # Strategy 2: top-degree seeds
    deg = [len(adj[i]) for i in range(n)]
    top = sorted(range(n), key=lambda x: -deg[x])
    for k in (4, 5, 6):
        if time_check():
            break
        grown = try_grow(list(top[:k]))
        update_best(grown, verify(grown))

    # Strategy 3: random seeds
    rng = _random.Random(12345)
    for _ in range(40):
        if time_check():
            break
        s = rng.randint(2, 5)
        grown = try_grow(rng.sample(range(n), s))
        update_best(grown, verify(grown))

    # Strategy 4: local search from best
    if best_branches is not None and not time_check():
        improved = local_search(best_branches)
        update_best(improved, verify(improved))

    # Strategy 5: grow from pairs of high-degree vertices
    if not time_check():
        high = [v for v in range(n) if deg[v] >= 5]
        for i in range(len(high)):
            if time_check():
                break
            for j in range(i + 1, len(high)):
                if time_check():
                    break
                grown = try_grow([high[i], high[j]])
                update_best(grown, verify(grown))

    # Fallback: at least 2 branches
    if best_branches is None or len(best_branches) < 2:
        u, v = edges[0]
        best_branches = [u, v]
        best_paths = {(u, v): [u, v]}

    # Ensure best_paths is consistent with best_branches
    if best_paths is None:
        best_paths = verify(best_branches)
    if best_paths is None:
        # last-resort: direct edges only
        best_paths = {}
        for u, v in combinations(sorted(best_branches), 2):
            if u in adj[v]:
                best_paths[(u, v)] = [u, v]
        if len(best_paths) != len(list(combinations(sorted(best_branches), 2))):
            # fall back to the first two vertices
            best_branches = [edges[0][0], edges[0][1]]
            best_paths = {(best_branches[0], best_branches[1]): list(best_branches)}

    branches = [int(b) for b in best_branches]
    paths_list = []
    for u, v in combinations(sorted(branches), 2):
        path = best_paths.get((u, v)) or best_paths.get((v, u))
        if path is None:
            path = [u, v]
        paths_list.append([int(x) for x in path])

    return {'branches': branches, 'paths': paths_list}
# EVOLVE-BLOCK-END
