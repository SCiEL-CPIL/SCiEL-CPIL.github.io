"""Improved: larger search pool, greedy+local search, optimized matching."""
import time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    degrees = [len(adj[i]) for i in range(n)]

    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    # Precompute common neighbours
    common = [[None] * n for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            c = sorted(adj[u] & adj[v])
            common[u][v] = c
            common[v][u] = c

    def check_and_build(B):
        """Given branch set B (list), try to find valid paths. Returns (branches, paths) or None."""
        if len(B) < 2:
            return None
        B_set = set(B)
        non_adj = []
        for i in range(len(B)):
            for j in range(i + 1, len(B)):
                u, v = B[i], B[j]
                if v not in adj[u]:
                    non_adj.append((u, v))

        # Pigeonhole
        if len(non_adj) > n - len(B):
            return None

        # For each non-adj pair, candidate internal vertices
        cands = []
        for u, v in non_adj:
            ws = [w for w in common[u][v] if w not in B_set]
            if not ws:
                return None
            cands.append(ws)

        # Hopcroft-Karp matching
        m = len(non_adj)
        if m == 0:
            # All pairs are adjacent - just use direct edges
            paths = []
            for i in range(len(B)):
                for j in range(i + 1, len(B)):
                    paths.append([B[i], B[j]])
            return (list(B), paths)

        right_verts = [v for v in range(n) if v not in B_set]
        right_idx = {v: i for i, v in enumerate(right_verts)}
        adj_left = []
        for ws in cands:
            adj_left.append([right_idx[w] for w in ws if w in right_idx])

        # Check if any left node has no edges
        for i in range(m):
            if not adj_left[i]:
                return None

        pair_u = [-1] * m
        pair_v = [-1] * len(right_verts)
        dist = [0] * m

        def bfs():
            q = deque()
            for u in range(m):
                if pair_u[u] == -1:
                    dist[u] = 0
                    q.append(u)
                else:
                    dist[u] = -1
            found = False
            while q:
                u = q.popleft()
                for v in adj_left[u]:
                    pu = pair_v[v]
                    if pu == -1:
                        found = True
                    elif dist[pu] == -1:
                        dist[pu] = dist[u] + 1
                        q.append(pu)
            return found

        def dfs(u):
            for v in adj_left[u]:
                pu = pair_v[v]
                if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                    pair_u[u] = v
                    pair_v[v] = u
                    return True
            dist[u] = -1
            return False

        matching = 0
        while bfs():
            for u in range(m):
                if pair_u[u] == -1 and dfs(u):
                    matching += 1
            if matching == m:
                break

        if matching < m:
            return None

        # Build paths
        paths = []
        for i in range(len(B)):
            for j in range(i + 1, len(B)):
                u, v = B[i], B[j]
                if v in adj[u]:
                    paths.append([u, v])
        for i, (u, v) in enumerate(non_adj):
            w = right_verts[pair_u[i]]
            paths.append([u, w, v])

        return (list(B), paths)

    # Strategy 1: Greedy construction starting from high-degree vertices
    sorted_vertices = sorted(range(n), key=lambda x: -degrees[x])

    def greedy_build():
        """Greedy: add vertices one by one if feasible."""
        B = [sorted_vertices[0]]
        for v in sorted_vertices[1:]:
            if time.time() > deadline:
                break
            new_B = B + [v]
            res = check_and_build(new_B)
            if res is not None:
                B = new_B
        return B

    best_B = greedy_build()
    best_result = check_and_build(best_B) if best_B else None

    # Strategy 2: Local search - try to add more vertices
    if best_B and time.time() < deadline:
        B_set = set(best_B)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding each non-branch vertex
            for v in range(n):
                if v in B_set or time.time() > deadline:
                    continue
                new_B = best_B + [v]
                res = check_and_build(new_B)
                if res is not None:
                    best_B = new_B
                    B_set = set(best_B)
                    best_result = res
                    improved = True
                    break

    # Strategy 3: Try to improve via swap (remove one, add two)
    if best_B and time.time() < deadline:
        B_set = set(best_B)
        for i in range(len(best_B)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(best_B)):
                if time.time() > deadline:
                    break
                # Try removing best_B[i] and best_B[j], adding two others
                remaining = [best_B[k] for k in range(len(best_B)) if k != i and k != j]
                for v1 in range(n):
                    if v1 in B_set or v1 == best_B[i] or v1 == best_B[j] or time.time() > deadline:
                        continue
                    for v2 in range(n):
                        if v2 in B_set or v2 == best_B[i] or v2 == best_B[j] or v2 == v1 or time.time() > deadline:
                            continue
                        new_B = remaining + [v1, v2]
                        res = check_and_build(new_B)
                        if res is not None and len(new_B) > len(best_B):
                            best_B = new_B
                            B_set = set(best_B)
                            best_result = res
                            break

    # Strategy 4: Systematic search for larger k
    # Try k from len(best_B)+1 upward with smart candidate selection
    current_best_k = len(best_B) if best_B else 0

    if time.time() < deadline:
        # Pool: all vertices, sorted by degree
        pool = sorted_vertices
        for k in range(current_best_k + 1, min(n, 16) + 1):
            if time.time() > deadline:
                break
            # Use a smaller pool for larger k to keep search tractable
            pool_size = min(len(pool), max(12, 25 - k))
            search_pool = pool[:pool_size]
            found = False
            for combo in combinations(search_pool, k):
                if time.time() > deadline:
                    found = True
                    break
                B = list(combo)
                res = check_and_build(B)
                if res is not None:
                    best_B = B
                    best_result = res
                    found = True
                    break
            if found:
                current_best_k = k
                # Continue trying to go higher
            else:
                # If we can't find k, maybe try a different pool ordering
                # Try with all vertices
                if pool_size < n and time.time() < deadline - 10:
                    search_pool = pool  # all vertices
                    found2 = False
                    count = 0
                    for combo in combinations(search_pool, k):
                        if time.time() > deadline:
                            found2 = True
                            break
                        count += 1
                        if count > 5000:  # limit
                            break
                        B = list(combo)
                        res = check_and_build(B)
                        if res is not None:
                            best_B = B
                            best_result = res
                            found2 = True
                            break
                    if found2:
                        current_best_k = k

    # Strategy 5: Max clique as additional check (all adjacent, no internal needed)
    if time.time() < deadline:
        def find_max_clique():
            best = []
            def bk(R, Pset, X):
                nonlocal best
                if not Pset and not X:
                    if len(R) > len(best):
                        best = list(R)
                    return
                if Pset:
                    pivot = max(Pset, key=lambda v: len(adj[v] & Pset))
                    cands = Pset - adj[pivot]
                else:
                    cands = Pset
                for v in list(cands):
                    bk(R | {v}, Pset & adj[v], X & adj[v])
                    Pset.remove(v)
                    X.add(v)
            bk(set(), set(range(n)), set())
            return best

        clique = find_max_clique()
        if len(clique) > (current_best_k if best_B else 0):
            B = sorted(clique)
            res = check_and_build(B)
            if res is not None:
                best_B = B
                best_result = res
                current_best_k = len(B)

    # Final: ensure we have at least 2 branches
    if best_result is None:
        # Fallback: find any valid pair
        for u, v in P['edges']:
            return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
        return {'branches': [0, 1], 'paths': [[0, 1]]}

    branches, paths = best_result
    return {'branches': [int(b) for b in branches], 'paths': [[int(x) for x in p] for p in paths]}
