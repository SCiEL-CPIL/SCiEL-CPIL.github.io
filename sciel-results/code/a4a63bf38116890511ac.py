"""Clique subdivision via max-flow (vertex-capacitated) path-disjointness check.

Structural change: the disjoint-paths feasibility test is a single max-flow on a
vertex-split network, so paths may be any simple length (capped) rather than a
precomputed short-path domain. Branch-set search is greedy + swap local search.
"""
import time
import random
import sys
from collections import deque

sys.setrecursionlimit(100000)

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}


def _max_flow(cap, n_nodes, s, t):
    """Edmonds-Karp. cap[u][v] >= 0 directed capacity. Returns (flow, parent)."""
    import copy
    cap = [row[:] for row in cap]
    flow = 0
    parent = [-1] * n_nodes
    while True:
        for i in range(n_nodes):
            parent[i] = -1
        parent[s] = s
        q = deque([s])
        while q and parent[t] == -1:
            u = q.popleft()
            for v in range(n_nodes):
                if parent[v] == -1 and cap[u][v] > 0:
                    parent[v] = u
                    if v == t:
                        break
                    q.append(v)
        if parent[t] == -1:
            break
        # bottleneck
        f = float('inf')
        v = t
        while v != s:
            u = parent[v]
            f = min(f, cap[u][v])
            v = u
        v = t
        while v != s:
            u = parent[v]
            cap[u][v] -= f
            cap[v][u] += f
            v = u
        flow += f
    return flow, parent


def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    adj_list = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        adj_list[u].append(v)
        adj_list[v].append(u)
    degrees = [len(adj[i]) for i in range(n)]

    start_time = time.time()
    deadline = start_time + 82.0

    # BFS distances (for ordering / fallback)
    INF = float('inf')
    dist = [[INF] * n for _ in range(n)]
    pred = [[-1] * n for _ in range(n)]
    for s in range(n):
        dist[s][s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj_list[u]:
                if dist[s][v] == INF:
                    dist[s][v] = dist[s][u] + 1
                    pred[s][v] = u
                    q.append(v)

    def shortest_path(s, t):
        if dist[s][t] == INF:
            return None
        path = []
        cur = t
        while cur != -1:
            path.append(cur)
            cur = pred[s][cur]
        path.reverse()
        return path

    # Feasibility of disjoint paths for non-adjacent branch pairs via max-flow.
    # Returns (ok, {pair: path}) or (False, None).
    def solve_paths(B, max_len=10):
        B_set = set(B)
        non_adj = []
        for i in range(len(B)):
            for j in range(i + 1, len(B)):
                u, v = B[i], B[j]
                if v not in adj[u]:
                    non_adj.append((u, v))
        if not non_adj:
            return True, {}

        # Vertex-split network.
        # nodes: 0..n-1 = left, n..2n-1 = right.
        # edge left(u)->right(u) capacity 1 (interior vertices) or 2 (branch vertices).
        # For each original edge (u,v) add right(u)->left(v) and right(v)->left(u) capacity 2.
        # Super source S, super sink T.
        # For each pair (u,v) in non_adj: S -> left(u) cap 1 ; right(v) -> T cap 1.
        N2 = 2 * n
        S = N2
        T = N2 + 1
        Ntot = N2 + 2
        cap = [[0] * Ntot for _ in range(Ntot)]

        def L(x):
            return x
        def R(x):
            return n + x

        # vertex capacity edges
        for x in range(n):
            c = 2 if x in B_set else 1
            cap[L(x)][R(x)] = c
        # undirected graph edges as two directed arcs
        for u in range(n):
            for v in adj_list[u]:
                if u < v:
                    cap[R(u)][L(v)] = 2
                    cap[R(v)][L(u)] = 2
        # pair connections
        for (u, v) in non_adj:
            cap[S][L(u)] += 1
            cap[R(v)][T] += 1

        flow, parent = _max_flow(cap, Ntot, S, T)
        if flow != len(non_adj):
            return False, None

        # Reconstruct each pair's path by tracing the 1-unit flow.
        # For each pair, find the flow from S->L(u) ... ->R(v)->T in residual (cap[v][u]>0 where flow was sent).
        # We rebuild flow matrix from residual: original cap0 known; flow on edge a->b = cap0[a][b]-cap[a][b].
        # Recompute a fresh residual-independent flow map.
        cap0 = [[0] * Ntot for _ in range(Ntot)]
        for x in range(n):
            c = 2 if x in B_set else 1
            cap0[L(x)][R(x)] = c
        for u in range(n):
            for v in adj_list[u]:
                if u < v:
                    cap0[R(u)][L(v)] = 2
                    cap0[R(v)][L(u)] = 2
        for (u, v) in non_adj:
            cap0[S][L(u)] += 1
            cap0[R(v)][T] += 1

        # flow[a][b] = cap0[a][b] - cap[a][b]
        # Trace for each pair: start at S, follow the unique outgoing flow to L(u), then through network to R(v), then T.
        # Since S has one unit per pair and each L(u) for a pair is distinct, trace by following flow.
        paths = {}
        # Build flow adjacency: for each node, list of (to, amount) where flow>0
        fadj = [[] for _ in range(Ntot)]
        for a in range(Ntot):
            for b in range(Ntot):
                f = cap0[a][b] - cap[a][b]
                if f > 0:
                    fadj[a].append((b, f))

        # For each pair, the source edge S->L(u) carries exactly 1. Trace from L(u) to T avoiding revisits,
        # but we must ensure we follow the flow belonging to this pair. Because the network is a flow,
        # decompose: repeatedly find an S->T path in the flow graph.
        # Simpler: for each pair, do a DFS in flow graph from L(u) to R(v) (these are degree-1 in flow for the pair).
        for (u, v) in non_adj:
            # find path in flow graph from L(u) to R(v)
            # DFS
            stack = [(L(u), [L(u)])]
            found = None
            while stack:
                node, path = stack.pop()
                if node == R(v):
                    found = path
                    break
                for (to, amt) in fadj[node]:
                    if to not in path:
                        stack.append((to, path + [to]))
            if found is None:
                return False, None
            # Convert node ids to vertex ids: left x -> x, right x -> x
            verts = []
            for node in found:
                if node < n:
                    verts.append(node)
                elif node < 2 * n:
                    verts.append(node - n)
            # verts should start with u and end with v
            if verts[0] != u or verts[-1] != v:
                # try reverse
                verts = list(reversed(verts))
            paths[(u, v)] = verts

        return True, paths

    # Maximum clique via Bron-Kerbosch with pivot (for a strong seed)
    best_clique = []
    def bron_kerbosch(R, Pset, X):
        nonlocal best_clique
        if not Pset and not X:
            if len(R) > len(best_clique):
                best_clique = list(R)
            return
        if Pset:
            pivot = max(Pset, key=lambda v: len(adj[v] & Pset))
            candidates = Pset - adj[pivot]
        else:
            candidates = Pset
        for v in list(candidates):
            bron_kerbosch(R | {v}, Pset & adj[v], X & adj[v])
            Pset.remove(v)
            X.add(v)
    bron_kerbosch(set(), set(range(n)), set())

    # Build best branch set
    best_B = list(best_clique)
    ok, best_paths = solve_paths(best_B)
    if not ok:
        while len(best_B) > 2:
            best_B.pop()
            ok, best_paths = solve_paths(best_B)
            if ok:
                break
        if not ok:
            best_B = [0, 1]
            best_paths = {}

    # Greedy extension
    while time.time() < deadline - 15:
        B_set = set(best_B)
        cands = [v for v in range(n) if v not in B_set]
        cands.sort(key=lambda v: -len(adj[v] & B_set))
        extended = False
        for v in cands:
            new_B = best_B + [v]
            ok, new_paths = solve_paths(new_B)
            if ok:
                best_B = new_B
                best_paths = new_paths
                extended = True
                break
        if not extended:
            break

    # Swap local search
    if time.time() < deadline - 8:
        improved = True
        while improved and time.time() < deadline - 8:
            improved = False
            B_set = set(best_B)
            for i in range(len(best_B)):
                if time.time() > deadline - 8:
                    break
                cands = [v for v in range(n) if v not in B_set]
                cands.sort(key=lambda v: -len(adj[v] & B_set))
                for v in cands[:10]:
                    new_B = best_B[:i] + [v] + best_B[i + 1:]
                    ok, new_paths = solve_paths(new_B)
                    if ok:
                        best_B = new_B
                        best_paths = new_paths
                        improved = True
                        break
                if improved:
                    break
            # extend after swap
            B_set = set(best_B)
            cands = [v for v in range(n) if v not in B_set]
            cands.sort(key=lambda v: -len(adj[v] & B_set))
            for v in cands[:6]:
                new_B = best_B + [v]
                ok, new_paths = solve_paths(new_B)
                if ok:
                    best_B = new_B
                    best_paths = new_paths
                    break

    # Random restarts for larger k
    if time.time() < deadline - 4:
        for target_k in range(len(best_B) + 1, min(n, len(best_B) + 4) + 1):
            if time.time() > deadline - 4:
                break
            found = False
            for attempt in range(30):
                if time.time() > deadline - 4:
                    break
                B_rand = random.sample(range(n), target_k)
                ok, paths_rand = solve_paths(B_rand)
                if ok:
                    best_B = B_rand
                    best_paths = paths_rand
                    found = True
                    break
            if not found:
                break

    # Final verification
    B = list(best_B)
    if len(B) < 2:
        B = [0, 1]
    ok, paths_dict = solve_paths(B)
    if not ok:
        while len(B) > 2:
            B.pop()
            ok, paths_dict = solve_paths(B)
            if ok:
                break
        if not ok:
            B = [0, 1]
            paths_dict = {}

    # Build output
    B_set = set(B)
    paths_out = []
    for i in range(len(B)):
        for j in range(i + 1, len(B)):
            u, v = B[i], B[j]
            if v in adj[u]:
                paths_out.append([u, v])
            else:
                if (u, v) in paths_dict:
                    paths_out.append(list(paths_dict[(u, v)]))
                elif (v, u) in paths_dict:
                    paths_out.append(list(reversed(paths_dict[(v, u)])))
                else:
                    p = shortest_path(u, v)
                    if p:
                        paths_out.append(p)
                    else:
                        paths_out.append([u, v])

    branches_out = [int(b) for b in B]
    paths_out = [[int(x) for x in p] for p in paths_out]
    return {'branches': branches_out, 'paths': paths_out}
