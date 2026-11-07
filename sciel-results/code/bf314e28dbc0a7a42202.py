import time
import random
import sys
from collections import deque

def construct():
    P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

    n = P['n']
    adj = [set() for _ in range(n)]
    adj_list = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        adj_list[u].append(v)
        adj_list[v].append(u)

    # All-pairs shortest path distances (BFS) for heuristic ordering
    dist = [[n + 1 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
        q = deque([i])
        while q:
            u = q.popleft()
            for v in adj_list[u]:
                if dist[i][v] > dist[i][u] + 1:
                    dist[i][v] = dist[i][u] + 1
                    q.append(v)

    start = time.time()
    deadline = start + 82.0

    def time_left():
        return deadline - time.time()

    def max_flow_feasible(branches):
        """
        Exact check: are there internally-disjoint simple paths for every pair of
        branches? Returns (True, paths_list) or (False, None).
        Model: split each internal vertex into in/out with capacity 1.
        For each branch, add super-source s_b and super-sink t_b with capacity 1
        edges into/out of the branch's in/out nodes. One max flow with all
        super-sources at node 0 and all super-sinks at node 1; feasible iff
        flow == C(k,2).
        """
        k = len(branches)
        if k < 2:
            return True, []
        C = k * (k - 1) // 2

        # Node IDs
        # 0 = global source, 1 = global sink
        # for each original vertex i: in(i) = 2 + 2*i, out(i) = 3 + 2*i
        # then super nodes: 2 + 2*n + b, 2 + 2*n + n + b  (but we can just use 0,1 for global)
        # Simpler: source = 0, sink = 1; branch super-source/sink are merged into 0 and 1
        # via unit-cap edges to/from the branch's in/out.
        N = 2 + 2 * n  # 0,1 are reserved; but we need real source/sink nodes too.
        # Use 0 as source, 1 as sink, then in(i)=2+2*i, out(i)=3+2*i.
        # Wait: 2+2*(n-1) and 3+2*(n-1) go up to 3+2n-2 = 2n+1. So N=2n+2 works with 0,1 free.

        N = 2 + 2 * n
        adj_f = [[] for _ in range(N)]
        cap = [[0] * N for _ in range(N)]
        rev = [[0] * N for _ in range(N)]
        edge_id = [[0] * N for _ in range(N)]
        eid = 0

        def add_edge(u, v, c):
            nonlocal eid
            adj_f[u].append(eid)
            cap[u][v] += c
            if edge_id[v][u] == 0 and u != v:
                rev[eid][v] = 1  # not really used
            eid += 1

        # Proper Dinic with edge list
        # Reset
        adj_f = [[] for _ in range(N)]
        cap = []
        for _ in range(N):
            cap.append([])
        to = []
        flow_edges = []  # (u,v, index_in_cap_row_u, reverse_index)
        # Simpler: store edges as list of [u,v,cap,rev_index]
        edges = []

        def add_edge(u, v, c):
            edges.append([u, v, c])
            adj_f[u].append(len(edges) - 1)
            edges.append([v, u, 0])
            adj_f[v].append(len(edges) - 1)

        # Internal vertex capacity: in(i) -> out(i) capacity 1
        for i in range(n):
            inn = 2 + 2 * i
            out = 3 + 2 * i
            add_edge(inn, out, 1)

        # Edges between vertices: out(u) -> in(v) capacity 1 (and out(v) -> in(u) for undirected)
        for u, v in P['edges']:
            add_edge(3 + 2 * u, 2 + 2 * v, 1)
            add_edge(3 + 2 * v, 2 + 2 * u, 1)

        # Source/sink connections for branches
        branch_set = set(branches)
        # super-source 0 -> in(b) cap 1; out(b) -> super-sink 1 cap 1
        for b in branches:
            add_edge(0, 2 + 2 * b, 1)
            add_edge(3 + 2 * b, 1, 1)

        # Dinic
        def bfs_level():
            level = [-1] * N
            level[0] = 0
            q = deque([0])
            while q:
                u = q.popleft()
                for ei in adj_f[u]:
                    if edges[ei][2] > 0:
                        v = edges[ei][1]
                        if level[v] < 0:
                            level[v] = level[u] + 1
                            q.append(v)
            return level

        def dfs(u, pushed, level, ptr):
            if u == 1:
                return pushed
            for i in range(ptr[u], len(adj_f[u])):
                ptr[u] = i
                ei = adj_f[u][i]
                v = edges[ei][1]
                if edges[ei][2] > 0 and level[v] == level[u] + 1:
                    tr = dfs(v, min(pushed, edges[ei][2]), level, ptr)
                    if tr > 0:
                        edges[ei][2] -= tr
                        edges[ei ^ 1][2] += tr
                        return tr
            return 0

        total = 0
        while True:
            level = bfs_level()
            if level[1] < 0:
                break
            ptr = [0] * N
            while True:
                pushed = dfs(0, 10**9, level, ptr)
                if pushed == 0:
                    break
                total += pushed
            if total >= C:
                break

        if total < C:
            return False, None

        # Extract paths. For each unit of flow, trace from super-sink 1 back to super-source 0.
        # For each branch b, it sends 1 unit out via out(b)->1 and receives 1 unit in via 0->in(b).
        # We need to pair each in-branch with an out-branch. The flow decomposition gives paths.
        # Walk each path: start at super-source 0, follow one unit of flow to super-sink 1.
        paths = []
        # We'll trace by following positive-flow edges. Each traversal consumes the flow.
        # Copy residual to work on.
        ecap = [e[2] for e in edges]
        # Work with original forward edges only: forward edges have even index, reverse odd.
        # We'll trace by DFS from 0 to 1 using only edges with ecap>0, marking used.
        # Simpler: repeatedly find an augmenting path in the current flow (using reverse edges to
        # "undo" and re-route is complex). Instead, decompose the flow by walking from each
        # super-source edge.
        #
        # Approach: For each branch b_in (in-node of b), there is exactly 1 unit entering in(b)
        # from super-source 0. Follow that unit forward through the network until we hit super-sink 1.
        # Record the sequence of original vertices.
        # We need to track which forward edge carries the flow from in(b).
        # Build flow map: for each original edge (u,v) in the residual, flow = original_cap - residual_cap
        # but we only added forward edges with cap 1 and reverse 0, so forward residual = 1 - flow.
        # For simplicity, since each forward edge has cap 1, flow on forward edge = 1 if residual 0 else 0.
        # But some forward edges may not carry flow. Let's build a flow graph.

        # Build directed flow edges with positive flow.
        # Original forward edges are at even indices (0,2,4,...) in 'edges'.
        flow_adj = [[] for _ in range(N)]
        for i in range(0, len(edges), 2):
            u, v, c = edges[i]
            # flow = c - residual
            residual = ecap[i]
            f = c - residual
            if f > 0:
                flow_adj[u].append((v, f))

        # Now trace C paths from 0 to 1.
        # For each unit, pick an outgoing edge from 0 with positive flow, follow to 1, remove flow.
        # Since all flows are 1 per edge and the structure is a unit-capacity network, this works.
        for _ in range(C):
            # BFS/DFS from 0 to 1 in flow_adj, removing as we go
            parent = [-1] * N
            parent_edge = [-1] * N
            parent[0] = 0
            q = deque([0])
            found = False
            while q:
                u = q.popleft()
                if u == 1:
                    found = True
                    break
                for idx, (v, f) in enumerate(flow_adj[u]):
                    if f > 0 and parent[v] == -1:
                        parent[v] = u
                        parent_edge[v] = idx
                        q.append(v)
            if not found:
                return False, None
            # Reconstruct
            path_nodes = []
            c = 1
            while c != 0:
                path_nodes.append(c)
                u = parent[c]
                idx = parent_edge[c]
                # remove 1 unit of flow from flow_adj[u][idx]
                flow_adj[u][idx] = (flow_adj[u][idx][0], flow_adj[u][idx][1] - 1)
                c = u
            path_nodes.reverse()  # now starts at 0
            # path_nodes[0] == 0, path_nodes[-1] == 1
            # Interior nodes are in/out nodes of original vertices
            # Convert to original vertex sequence
            orig = []
            for node in path_nodes:
                if node == 0 or node == 1:
                    continue
                if node % 2 == 0:
                    # in-node: node = 2 + 2*i  => i = (node-2)//2
                    orig.append((node - 2) // 2)
                else:
                    # out-node: node = 3 + 2*i => i = (node-3)//2
                    orig.append((node - 3) // 2)
            # orig may have consecutive duplicates (in and out of same vertex collapse to same i)
            # Deduplicate consecutive
            dedup = []
            for x in orig:
                if not dedup or dedup[-1] != x:
                    dedup.append(x)
            # Validate: first and last must be branches, intermediates not branches
            if len(dedup) < 2:
                return False, None
            # We don't strictly need to match pair order; just return as a path.
            paths.append(dedup)

        return True, paths

    # Greedy build using flow oracle
    def build_greedy(seed_list, time_budget=2.0):
        branches = list(seed_list)
        t_end = time.time() + time_budget
        while time.time() < t_end:
            candidates = [v for v in range(n) if v not in branches]
            if not candidates:
                break
            # Score: prefer vertices adjacent to many existing branches and close
            def sc(v):
                cnt = sum(1 for b in branches if v in adj[b])
                md = min(dist[v][b] for b in branches) if branches else 0
                return (cnt, -md)
            candidates.sort(key=sc, reverse=True)
            for v in candidates:
                nb = branches + [v]
                ok, _ = max_flow_feasible(nb)
                if ok:
                    branches = nb
                    break
            else:
                break
        return branches

    # Local search with restarts
    def local_search(start_branches, time_budget=3.0):
        t_end = time.time() + time_budget
        current = list(start_branches)
        best = list(current)
        best_size = len(current)
        ok, _ = max_flow_feasible(current)
        if not ok:
            return None

        while time.time() < t_end:
            improved = False
            # Try adding
            candidates = [v for v in range(n) if v not in current]
            def sc_add(v):
                cnt = sum(1 for b in current if v in adj[b])
                md = min(dist[v][b] for b in current) if current else 0
                return (cnt, -md)
            candidates.sort(key=sc_add, reverse=True)
            for v in candidates:
                if time.time() >= t_end:
                    break
                nb = current + [v]
                ok2, _ = max_flow_feasible(nb)
                if ok2:
                    current = nb
                    if len(current) > best_size:
                        best = list(current)
                        best_size = len(current)
                    improved = True
                    break
            if improved:
                continue
            # Try swap: remove one, add another
            swapped = False
            # Random order of removals for diversity
            rem_order = list(range(len(current)))
            random.shuffle(rem_order)
            add_order = [v for v in range(n) if v not in current]
            random.shuffle(add_order)
            for ri in rem_order:
                if time.time() >= t_end:
                    break
                for v in add_order:
                    if time.time() >= t_end:
                        break
                    nb = [current[j] for j in range(len(current)) if j != ri] + [v]
                    if len(nb) < len(best):
                        continue
                    ok2, _ = max_flow_feasible(nb)
                    if ok2:
                        current = nb
                        if len(current) > best_size:
                            best = list(current)
                            best_size = len(current)
                        swapped = True
                        break
                if swapped:
                    break
            if not swapped:
                break
        return best

    # Seed set: find a small clique (or just use high-degree vertices)
    # Use Bron-Kerbosch to find a max clique as a strong seed
    def find_max_clique():
        best_c = []
        def bronk(R, P_set, X):
            nonlocal best_c
            if time.time() > deadline - 15:
                return
            if not P_set and not X:
                if len(R) > len(best_c):
                    best_c = list(R)
                return
            if P_set or X:
                S = P_set | X
                pivot = max(S, key=lambda v: len(adj[v] & P_set))
            else:
                return
            cand = [v for v in P_set if v not in adj[pivot]]
            for v in cand:
                bronk(R | {v}, P_set & adj[v], X & adj[v])
                P_set = P_set - {v}
                X = X | {v}
        bronk(frozenset(), frozenset(range(n)), frozenset())
        return best_c

    # Try several seeds: max clique, top-degree, random
    best_branches = None
    best_paths = None
    best_size = 0

    def try_seed(seed, label):
        nonlocal best_branches, best_paths, best_size
        if time.time() > deadline - 10:
            return
        # Build greedily
        built = build_greedy(seed, time_budget=1.5)
        if len(built) >= 2:
            ok, paths = max_flow_feasible(built)
            if ok and len(built) > best_size:
                best_branches = built
                best_paths = paths
                best_size = len(built)
        # Local search
        if time.time() > deadline - 8:
            return
        improved = local_search(built, time_budget=2.0)
        if improved and len(improved) > best_size:
            ok, paths = max_flow_feasible(improved)
            if ok:
                best_branches = improved
                best_paths = paths
                best_size = len(improved)

    # Seed 1: max clique
    clique = find_max_clique()
    if len(clique) >= 2:
        try_seed(clique, "clique")
    else:
        # fallback: two highest degree vertices
        order = sorted(range(n), key=lambda v: len(adj[v]), reverse=True)
        try_seed(order[:2], "top2")

    # Seed 2: top-degree vertices (up to 5)
    order = sorted(range(n), key=lambda v: len(adj[v]), reverse=True)
    if time.time() < deadline - 15:
        topk = order[:min(5, n)]
        try_seed(topk, "topdeg")

    # Seed 3: random sets
    for it in range(8):
        if time.time() > deadline - 12:
            break
        seed = random.sample(range(n), min(4, n))
        try_seed(seed, "rand")

    # Final 2-opt refinement on best
    if best_branches and time.time() < deadline - 6:
        # Try swapping pairs
        cur = list(best_branches)
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            improved = False
            rem = list(range(len(cur)))
            random.shuffle(rem)
            adds = [v for v in range(n) if v not in cur]
            random.shuffle(adds)
            for ri in rem:
                if time.time() > deadline - 5:
                    break
                for v in adds:
                    if time.time() > deadline - 5:
                        break
                    nb = [cur[j] for j in range(len(cur)) if j != ri] + [v]
                    if len(nb) < best_size:
                        continue
                    ok, _ = max_flow_feasible(nb)
                    if ok:
                        cur = nb
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        # Try adding more
        while time.time() < deadline - 4:
            adds = [v for v in range(n) if v not in cur]
            if not adds:
                break
            adds.sort(key=lambda v: sum(1 for b in cur if v in adj[b]), reverse=True)
            prog = False
            for v in adds:
                nb = cur + [v]
                ok, _ = max_flow_feasible(nb)
                if ok:
                    cur = nb
                    prog = True
                    break
            if not prog:
                break
        ok, paths = max_flow_feasible(cur)
        if ok and len(cur) > best_size:
            best_branches = cur
            best_paths = paths
            best_size = len(cur)

    # Return
    if best_branches and len(best_branches) >= 2 and best_paths is not None:
        branches = [int(b) for b in best_branches]
        formatted = []
        for p in best_paths:
            formatted.append([int(x) for x in p])
        return {'branches': branches, 'paths': formatted}

    # Fallback
    if P['edges']:
        u, v = P['edges'][0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
    return {'branches': [0, 1], 'paths': [[0, 1]]}
