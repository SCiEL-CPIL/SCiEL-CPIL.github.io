import time
import random
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin under 90s limit

    n = P['n']
    edges = P['edges']

    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    degrees = [len(adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda x: degrees[x], reverse=True)

    # Precompute all simple paths up to a certain length between every pair.
    # We'll store for each (u,v) a list of candidate paths (as tuples), shortest first.
    # To keep the search tractable, we limit path length to something reasonable.
    MAX_PATH_LEN = 8  # max number of vertices in a path (including endpoints)

    def all_simple_paths(s, t, max_len):
        """Return all simple paths from s to t with at most max_len vertices."""
        if s == t:
            return []
        results = []
        visited = [False] * n
        visited[s] = True

        def dfs(cur, path):
            if len(path) > max_len:
                return
            if cur == t and len(path) >= 2:
                results.append(tuple(path))
                return
            for nb in sorted(adj[cur]):
                if not visited[nb]:
                    visited[nb] = True
                    path.append(nb)
                    dfs(nb, path)
                    path.pop()
                    visited[nb] = False

        dfs(s, [s])
        return results

    # Cache: paths_db[(u,v)] = list of (path_tuple, interior_frozenset)
    paths_db = {}
    def get_candidate_paths(u, v):
        key = (min(u, v), max(u, v))
        if key not in paths_db:
            ps = all_simple_paths(u, v, MAX_PATH_LEN)
            # Sort by length
            ps.sort(key=len)
            # Deduplicate by interior set, keeping shortest
            seen = set()
            unique = []
            for p in ps:
                interior = frozenset(p[1:-1])
                if interior not in seen:
                    seen.add(interior)
                    unique.append((p, interior))
            paths_db[key] = unique
        return paths_db[key]

    # For a given branch set, solve the path assignment problem using branch-and-bound.
    # Each pair needs one path; paths must have disjoint interiors.
    def solve_path_assignment(branches):
        branches_set = set(branches)
        pairs = list(combinations(branches, 2))
        m = len(pairs)
        if m == 0:
            return []

        # Get candidate paths for each pair
        cands = []
        for (u, v) in pairs:
            ps = get_candidate_paths(u, v)
            if not ps:
                return None
            cands.append(ps)

        used = [0] * n  # used[v] = 1 if interior vertex v is used
        assign = [None] * m

        # Order pairs by number of candidates (most constrained first)
        order_pairs = sorted(range(m), key=lambda i: len(cands[i]))

        def rec(idx):
            if time.time() > deadline:
                return False
            if idx == len(order_pairs):
                return True
            i = order_pairs[idx]
            for p, interior in cands[i]:
                ok = True
                for v in interior:
                    if used[v]:
                        ok = False
                        break
                if not ok:
                    continue
                for v in interior:
                    used[v] = 1
                assign[i] = p
                if rec(idx + 1):
                    return True
                assign[i] = None
                for v in interior:
                    used[v] = 0
            return False

        if rec(0):
            return [list(assign[i]) for i in range(m)]
        return None

    best = None
    best_size = -1

    def consider(branches):
        nonlocal best, best_size
        if time.time() > deadline:
            return
        if len(branches) <= best_size:
            return
        paths = solve_path_assignment(list(branches))
        if paths is not None:
            best = {
                'branches': [int(b) for b in branches],
                'paths': [list(map(int, p)) for p in paths],
            }
            best_size = len(branches)

    # Strategy: find large cliques in the graph, then try to extend them.
    # Also try random sets of high-degree vertices.

    # Bron-Kerbosch to find all maximal cliques in a pool
    def all_maximal_cliques(pool, cap=500):
        pool_set = set(pool)
        found = []
        def bk(R, P, X):
            nonlocal found
            if time.time() > deadline:
                return
            if not P and not X:
                if R:
                    found.append(list(R))
                return
            if len(found) >= cap:
                return
            if P or X:
                pivot = max(list(P) + list(X), key=lambda v: len(adj[v] & P))
                cands = [v for v in P if v not in adj[pivot]]
            else:
                cands = list(P)
            for v in cands:
                bk(R + [v], P & adj[v], X & adj[v])
                P.discard(v)
                X.add(v)
        bk([], pool_set, set())
        return found

    # Phase 1: Find maximal cliques in the full graph
    cliques = all_maximal_cliques(range(n), cap=300)
    cliques.sort(key=len, reverse=True)
    for c in cliques:
        consider(c)
        if best_size >= 10:
            break

    # Phase 2: Try to extend cliques by adding non-adjacent vertices
    for c in cliques:
        if time.time() > deadline:
            break
        if len(c) + (n - len(c)) <= best_size:
            continue
        # Try adding one vertex at a time
        c_set = set(c)
        candidates = [v for v in range(n) if v not in c_set]
        for v in candidates:
            new_c = c + [v]
            if len(new_c) <= best_size:
                continue
            consider(new_c)
        # Try adding two vertices
        if time.time() > deadline:
            break
        for v in candidates:
            if time.time() > deadline:
                break
            for w in candidates:
                if w <= v:
                    continue
                new_c = c + [v, w]
                if len(new_c) <= best_size:
                    continue
                consider(new_c)

    # Phase 3: Random search with high-degree bias
    random.seed(42)
    for restart in range(200):
        if time.time() > deadline:
            break
        # Pick a random size between best_size+1 and min(n, best_size+4)
        target_size = min(n, best_size + 1 + random.randint(0, 3))
        if target_size <= best_size:
            target_size = best_size + 1
        if target_size < 2:
            target_size = 2

        # Sample vertices with bias toward high degree
        pool = order[:max(10, target_size + 5)]
        chosen = set(random.sample(pool, min(len(pool), target_size)))
        # Fill remaining with random vertices
        remaining = [v for v in range(n) if v not in chosen]
        while len(chosen) < target_size and remaining:
            chosen.add(remaining.pop(random.randint(0, len(remaining) - 1)))

        consider(list(chosen))

    # Phase 4: Systematic search for larger sets
    # Try all combinations of size best_size+1 from top-degree vertices
    if best_size < n:
        top_k = min(n, best_size + 6)
        top_vertices = order[:top_k]
        target = best_size + 1
        if target >= 2:
            for combo in combinations(top_vertices, target):
                if time.time() > deadline:
                    break
                consider(list(combo))

    if best is not None:
        return best

    # Fallback: max clique of whole graph (trivial subdivision)
    mc = cliques[0] if cliques else [order[0], order[1]]
    if len(mc) >= 2:
        branches = [int(v) for v in mc]
        paths = []
        for u, v in combinations(branches, 2):
            ps = get_candidate_paths(u, v)
            if ps:
                paths.append(list(ps[0][0]))
            else:
                # Direct edge check
                if v in adj[u]:
                    paths.append([int(u), int(v)])
                else:
                    # Should not happen for a clique
                    paths.append([int(u), int(v)])
        return {'branches': branches, 'paths': paths}

    # Absolute fallback
    u, v = edges[0]
    return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
