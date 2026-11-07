"""Solver for large clique subdivisions in a pinned graph."""

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}


def construct():
    import time
    from collections import deque
    from itertools import combinations

    t_start = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    adj_list = [list() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    for i in range(n):
        adj_list[i] = sorted(adj[i])
    deg = [len(adj[i]) for i in range(n)]

    def time_up():
        return time.time() - t_start > DEADLINE

    # Precompute all-pairs shortest paths for ordering heuristics
    sp_dist = [[0] * n for _ in range(n)]
    for s in range(n):
        dist = [10**9] * n
        dist[s] = 0
        q = deque([s])
        while q:
            cur = q.popleft()
            for nb in adj_list[cur]:
                if dist[nb] > dist[cur] + 1:
                    dist[nb] = dist[cur] + 1
                    q.append(nb)
        sp_dist[s] = dist

    def bfs_path(src, dst, forbidden):
        """BFS simple path src->dst; interior vertices avoid forbidden."""
        if src == dst:
            return [src]
        parent = {src: -1}
        q = deque([src])
        while q:
            cur = q.popleft()
            if cur == dst:
                path = []
                node = dst
                while node != -1:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return path
            for nb in adj_list[cur]:
                if nb not in parent:
                    if nb != dst and nb in forbidden:
                        continue
                    parent[nb] = cur
                    q.append(nb)
        return None

    def find_alternatives(u, v, used, bset, max_alts=4):
        """Find up to max_alts distinct simple paths u->v avoiding used interiors."""
        alts = []
        p0 = bfs_path(u, v, used)
        if p0 is None:
            return alts
        alts.append(tuple(p0))
        seen = {tuple(p0)}
        # Try excluding each interior vertex of p0 to force different routes
        for mid in p0[1:-1]:
            if len(alts) >= max_alts:
                break
            alt = bfs_path(u, v, used | {mid})
            if alt is not None:
                t = tuple(alt)
                if t not in seen:
                    seen.add(t)
                    alts.append(t)
        # Try excluding pairs of interior vertices
        if len(alts) < max_alts:
            for a in p0[1:-1]:
                for b in p0[1:-1]:
                    if a == b:
                        continue
                    if len(alts) >= max_alts:
                        break
                    alt = bfs_path(u, v, used | {a, b})
                    if alt is not None:
                        t = tuple(alt)
                        if t not in seen:
                            seen.add(t)
                            alts.append(t)
        return alts

    def solve_paths(branches, order, max_alts=4):
        """Backtracking: assign internally-disjoint paths for all pairs in order.
        Returns dict {(u,v): path} or None."""
        bset = set(branches)

        def bt(idx, used, paths):
            if time_up():
                return None
            if idx == len(order):
                return paths
            u, v = order[idx]
            alts = find_alternatives(u, v, used, bset, max_alts)
            if not alts:
                return None
            for path_t in alts:
                interior = set(path_t[1:-1])
                if interior & bset or interior & used:
                    continue
                new_used = used | interior
                paths[(min(u, v), max(u, v))] = list(path_t)
                result = bt(idx + 1, new_used, paths)
                if result is not None:
                    return result
                del paths[(min(u, v), max(u, v))]
            return None

        return bt(0, set(), {})

    def make_order(branches, mode):
        """Generate a pair ordering for the given branch set."""
        pairs = list(combinations(branches, 2))
        if mode == 0:
            # direct edges first, then by distance
            pairs.sort(key=lambda p: (0 if p[1] in adj[p[0]] else 1,
                                       sp_dist[p[0]][p[1]], p[0], p[1]))
        elif mode == 1:
            # high degree sum first
            pairs.sort(key=lambda p: -(deg[p[0]] + deg[p[1]]))
        elif mode == 2:
            # low degree first (constrained endpoints)
            pairs.sort(key=lambda p: (deg[p[0]], deg[p[1]]))
        elif mode == 3:
            # reverse of mode 0
            pairs.sort(key=lambda p: (0 if p[1] in adj[p[0]] else 1,
                                       sp_dist[p[0]][p[1]], p[0], p[1]))
            pairs.reverse()
        elif mode == 4:
            # by distance only
            pairs.sort(key=lambda p: (sp_dist[p[0]][p[1]], p[0], p[1]))
        else:
            pairs.sort(key=lambda p: (p[0], p[1]))
        return pairs

    def make_result(branches, paths):
        bset = set(branches)
        path_list = []
        for u, v in combinations(branches, 2):
            key = (min(u, v), max(u, v))
            path_list.append(list(paths[key]))
        return {'branches': list(branches), 'paths': path_list}

    def try_branch_set(branches, max_alts=4):
        """Try multiple orderings; return result dict or None."""
        for mode in range(5):
            if time_up():
                return None
            order = make_order(branches, mode)
            paths = solve_paths(branches, order, max_alts)
            if paths is not None:
                return make_result(branches, paths)
        return None

    # --- Find cliques of various sizes as seeds ---
    def find_cliques(max_size=8, limit=200):
        """Find cliques via greedy + local search."""
        cliques = []
        # Greedy: start from each high-degree vertex
        candidates = sorted(range(n), key=lambda i: (-deg[i], i))
        for start in candidates:
            if time_up():
                break
            clique = [start]
            for v in candidates:
                if v == start:
                    continue
                if all(v in adj[w] for w in clique):
                    clique.append(v)
                if len(clique) >= max_size:
                    break
            if len(clique) >= 2:
                cliques.append(clique)
            if len(cliques) >= limit:
                break
        # Also try random-ish starts
        for start in range(n):
            if start in [c[0] for c in cliques]:
                continue
            clique = [start]
            for v in candidates:
                if v == start:
                    continue
                if all(v in adj[w] for w in clique):
                    clique.append(v)
                if len(clique) >= max_size:
                    break
            if len(clique) >= 2:
                cliques.append(clique)
            if len(cliques) >= limit * 2:
                break
        return cliques

    best = None
    best_size = 0

    # --- Phase 1: Greedy growth from multiple seeds ---
    cliques = find_cliques(max_size=8, limit=100)

    # Sort seeds: larger first, then by sum of degrees
    cliques.sort(key=lambda c: (-len(c), -sum(deg[v] for v in c)))

    # Candidate pool for growth: all vertices, ordered by degree
    growth_pool = sorted(range(n), key=lambda i: (-deg[i], i))

    for seed in cliques:
        if time_up():
            break
        if len(seed) <= best_size:
            continue
        # Try growing this seed
        branches = list(seed)
        # First verify the seed itself
        result = try_branch_set(branches, max_alts=4)
        if result is None:
            # Seed not valid as-is; try smaller subsets
            for sz in range(len(seed) - 1, 1, -1):
                if sz <= best_size:
                    break
                for sub in combinations(seed, sz):
                    if time_up():
                        break
                    result = try_branch_set(list(sub), max_alts=4)
                    if result is not None:
                        branches = list(sub)
                        break
                if result is not None:
                    break
            if result is None:
                continue
            best = result
            best_size = len(branches)
            if len(branches) <= best_size:
                continue

        # Now grow: try adding each candidate branch
        changed = True
        while changed and not time_up():
            changed = False
            # Try candidates in degree order
            for cand in growth_pool:
                if cand in branches:
                    continue
                if len(branches) + 1 <= best_size:
                    break
                new_branches = branches + [cand]
                result = try_branch_set(new_branches, max_alts=5)
                if result is not None:
                    branches = new_branches
                    best = result
                    best_size = len(branches)
                    changed = True
                    break
            # Also try a few lower-degree candidates
            for cand in growth_pool:
                if cand in branches:
                    continue
                if deg[cand] < 4:
                    continue
                if len(branches) + 1 <= best_size:
                    break
                new_branches = branches + [cand]
                result = try_branch_set(new_branches, max_alts=5)
                if result is not None:
                    branches = new_branches
                    best = result
                    best_size = len(branches)
                    changed = True
                    break

    # --- Phase 2: Top-down search for larger sets if growth didn't find big ones ---
    if best_size < 8:
        candidates = sorted(range(n), key=lambda i: (-deg[i], i))
        for size in range(min(n, 12), best_size, -1):
            if time_up():
                break
            top_k = min(n, size + 6)
            pool = candidates[:top_k]
            tried = 0
            max_tries = 3000
            for combo in combinations(pool, size):
                if time_up() or tried >= max_tries:
                    break
                tried += 1
                result = try_branch_set(list(combo), max_alts=5)
                if result is not None:
                    best = result
                    best_size = size
                    break
            if best_size >= size:
                break

    # --- Phase 3: If still small, try backtracking on promising sets ---
    if best_size < 6:
        for size in range(min(n, 9), max(best_size, 3), -1):
            if time_up():
                break
            if size <= best_size:
                break
            top_k = min(n, size + 5)
            pool = candidates[:top_k]
            tried = 0
            max_tries = 1500
            for combo in combinations(pool, size):
                if time_up() or tried >= max_tries:
                    break
                tried += 1
                branches = list(combo)
                # Use more alternatives for backtracking
                for mode in range(5):
                    if time_up():
                        break
                    order = make_order(branches, mode)
                    paths = solve_paths(branches, order, max_alts=6)
                    if paths is not None:
                        best = make_result(branches, paths)
                        best_size = size
                        break
                if best_size >= size:
                    break
            if best_size >= size:
                break

    # --- Fallback: find a triangle ---
    if best is None:
        for a, b, c in combinations(range(n), 3):
            if b in adj[a] and c in adj[a] and c in adj[b]:
                best = {
                    'branches': [a, b, c],
                    'paths': [[a, b], [a, c], [b, c]]
                }
                break
        if best is None:
            u, v = edges[0]
            best = {'branches': [u, v], 'paths': [[u, v]]}

    # Convert to proper types
    branches = [int(x) for x in best['branches']]
    paths = [[int(x) for x in p] for p in best['paths']]
    return {'branches': branches, 'paths': paths}
