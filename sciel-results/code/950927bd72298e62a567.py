import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = set()
        for i in range(n - 1):
            u, v = order[i], order[i + 1]
            edges.add((min(u, v), max(u, v)))
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return list(edges)

    def hopcroft_karp(n_left, n_right, adj):
        INF = float('inf')
        pair_u = [-1] * n_left
        pair_v = [-1] * n_right
        dist = [0] * n_left

        def bfs():
            queue = deque()
            for u in range(n_left):
                if pair_u[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = INF
            found = False
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    pv = pair_v[v]
                    if pv == -1:
                        found = True
                    elif dist[pv] == INF:
                        dist[pv] = dist[u] + 1
                        queue.append(pv)
            return found

        def dfs(u):
            for v in adj[u]:
                pv = pair_v[v]
                if pv == -1 or (dist[pv] == dist[u] + 1 and dfs(pv)):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            dist[u] = INF
            return False

        while bfs():
            for u in range(n_left):
                if pair_u[u] == -1:
                    dfs(u)
        return pair_u

    def build_paths(edge_list, succ):
        m = len(edge_list)
        pred = {}
        for i in range(m):
            j = succ.get(i)
            if j is not None:
                pred[j] = i
        starts = [i for i in range(m) if i not in pred]
        paths = []
        for st in starts:
            path_edges = []
            cur = st
            seen = set()
            while cur is not None and cur not in seen:
                seen.add(cur)
                path_edges.append(cur)
                cur = succ.get(cur)
            if not path_edges:
                continue
            vp = []
            for ei in path_edges:
                u, v = edge_list[ei]
                if not vp:
                    vp = [u, v]
                else:
                    last = vp[-1]
                    if last == u:
                        vp.append(v)
                    elif last == v:
                        vp.append(u)
                    else:
                        vp.append(u)
            paths.append(vp)
        # Any edge not in a path (shouldn't happen) -> add as singleton
        used = set()
        for vp in paths:
            for k in range(len(vp) - 1):
                a, b = vp[k], vp[k+1]
                used.add((min(a,b), max(a,b)))
        for i, e in enumerate(edge_list):
            if e not in used:
                paths.append(list(e))
                used.add(e)
        return paths

    def validate_and_fix(edge_set, paths):
        # Ensure each path is simple and edges valid & edge-disjoint
        used = set()
        ok_paths = []
        for vp in paths:
            newp = []
            ok = True
            for k in range(len(vp)):
                vtx = vp[k]
                if newp and vtx == newp[-1]:
                    continue
                if not newp:
                    newp.append(vtx)
                else:
                    a, b = newp[-1], vtx
                    e = (min(a,b), max(a,b))
                    if e not in edge_set or e in used:
                        ok = False
                        break
                    used.add(e)
                    newp.append(vtx)
            if ok and newp:
                ok_paths.append(newp)
        # Add any uncovered edges as singletons
        for e in edge_set:
            if e not in used:
                ok_paths.append(list(e))
                used.add(e)
        return ok_paths

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    all_results = []

    for spec in specs:
        n, p, seed = spec
        edges = instance(n, p, seed)
        m = len(edges)
        edge_set = set(edges)

        if m == 0:
            all_results.append([])
            continue

        edge_list = list(edges)

        # Build edge-vertex incidence
        edge_verts = [set(edge_list[i]) for i in range(m)]

        # Left adjacency: edge i can precede edge j if they share a vertex
        left_adj = [[] for _ in range(m)]
        for i in range(m):
            vi = edge_verts[i]
            for j in range(m):
                if i == j:
                    continue
                if vi & edge_verts[j]:
                    left_adj[i].append(j)

        pair_u = hopcroft_karp(m, m, left_adj)
        succ = {i: pair_u[i] for i in range(m) if pair_u[i] != -1}
        paths = build_paths(edge_list, succ)
        paths = validate_and_fix(edge_set, paths)
        all_results.append(paths)

    # Local improvement: try to merge two paths if they can be concatenated
    # (end of one adjacent to start of another) and result stays simple.
    for idx in range(len(all_results)):
        paths = all_results[idx]
        edge_set = None
        # rebuild edge set from paths
        all_edges = set()
        for vp in paths:
            for k in range(len(vp) - 1):
                a, b = vp[k], vp[k+1]
                all_edges.add((min(a,b), max(a,b)))
        edge_set = all_edges

        improved = True
        while improved and (time.time() < deadline - 5.0):
            improved = False
            # try merging path A into path B at each end
            new_paths = list(paths)
            best = None
            for ia in range(len(new_paths)):
                for ib in range(len(new_paths)):
                    if ia == ib:
                        continue
                    A = new_paths[ia]
                    B = new_paths[ib]
                    # try B + A (concatenate)
                    for combo, newp in _merge_candidates(A, B):
                        if _is_valid_merge(newp, edge_set, B, A):
                            if best is None or len(newp) < len(best):
                                best = (ia, ib, newp)
                    if best is not None:
                        break
                if best is not None:
                    break
            if best is not None:
                ia, ib, newp = best
                # remove ia and ib, add newp
                new_paths = [p for k, p in enumerate(new_paths) if k != ia and k != ib]
                new_paths.append(newp)
                paths = new_paths
                improved = True
        all_results[idx] = paths

    return all_results


def _merge_candidates(A, B):
    # yield candidate concatenations
    # B followed by A: B_end adjacent to A_start
    if B and A:
        cand = B + [A[0]]
        yield cand, cand
    # A followed by B
    if A and B:
        cand = A + [B[0]]
        yield cand, cand


def _is_valid_merge(newp, edge_set, A, B):
    # newp must be simple, all edges in edge_set, and use exactly edges of A and B
    if len(newp) < 2:
        return False
    seen = set()
    for v in newp:
        if v in seen:
            return False
        seen.add(v)
    used = set()
    for k in range(len(newp) - 1):
        a, b = newp[k], newp[k+1]
        e = (min(a,b), max(a,b))
        if e not in edge_set or e in used:
            return False
        used.add(e)
    # used must equal edges of A and B
    target = set()
    for vp in (A, B):
        for k in range(len(vp) - 1):
            a, b = vp[k], vp[k+1]
            target.add((min(a,b), max(a,b)))
    return used == target
