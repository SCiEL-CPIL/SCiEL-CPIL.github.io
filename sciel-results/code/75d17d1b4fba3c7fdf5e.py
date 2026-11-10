"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}


def construct():
    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    deadline = _time.time() + 55  # safety margin

    def find_shortest_path(src, dst, forbidden):
        """Find shortest path from src to dst avoiding forbidden vertices (except src/dst)."""
        if src == dst:
            return [src]
        visited = {src}
        parent = {src: None}
        q = deque([src])
        while q:
            u = q.popleft()
            if u == dst:
                path = []
                cur = dst
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                return list(reversed(path))
            for v in adj[u]:
                if v not in visited and (v == dst or v not in forbidden):
                    visited.add(v)
                    parent[v] = u
                    q.append(v)
        return None

    def check_branch_set(branches):
        """
        Check if a set of branches has valid internally-disjoint paths between every pair.
        Uses a flow-based approach: model as a vertex-disjoint paths problem.
        Returns dict {(u,v): path} or None.
        """
        bset = set(branches)
        pairs = list(combinations(sorted(branches), 2))
        if not pairs:
            return {}

        # We'll try to find internally vertex-disjoint paths for all pairs.
        # Strategy: iteratively assign paths, trying multiple orderings.
        # Use BFS with increasing path length preference.

        best_paths = None
        best_total_interior = float('inf')

        # Precompute shortest path lengths between all pairs avoiding other branches
        sp_len = {}
        for u, v in pairs:
            forbidden = bset - {u, v}
            path = find_shortest_path(u, v, forbidden)
            if path is None:
                return None
            sp_len[(u, v)] = len(path)

        # Try several orderings
        import random
        rng = random.Random(99999)

        def try_ordering(ordering):
            used_interiors = set()
            paths = {}
            for u, v in ordering:
                forbidden = (bset - {u, v}) | used_interiors
                path = find_shortest_path(u, v, forbidden)
                if path is None:
                    return None
                interior = set(path[1:-1])
                if interior & bset:
                    return None
                if interior & used_interiors:
                    return None
                used_interiors |= interior
                paths[(u, v)] = path
            return paths

        # Ordering by shortest path length ascending
        order1 = sorted(pairs, key=lambda p: sp_len[p])
        p = try_ordering(order1)
        if p is not None:
            total = sum(len(path[1:-1]) for path in p.values())
            if total < best_total_interior:
                best_total_interior = total
                best_paths = p

        # Ordering by shortest path length descending
        order2 = sorted(pairs, key=lambda p: -sp_len[p])
        p = try_ordering(order2)
        if p is not None:
            total = sum(len(path[1:-1]) for path in p.values())
            if total < best_total_interior:
                best_total_interior = total
                best_paths = p

        # Random orderings
        for _ in range(20):
            if _time.time() > deadline:
                break
            shuffled = list(pairs)
            rng.shuffle(shuffled)
            p = try_ordering(shuffled)
            if p is not None:
                total = sum(len(path[1:-1]) for path in p.values())
                if total < best_total_interior:
                    best_total_interior = total
                    best_paths = p

        # Also try: for each pair, try to find the shortest path that doesn't conflict
        # with already-assigned paths, using a more careful approach
        # Greedy with backtracking-lite: try pairs in order, and if a pair fails,
        # try reassigning a previous pair

        if best_paths is not None:
            return best_paths
        return None

    def try_grow(initial_branches):
        """Try to grow a branch set greedily, trying to add one vertex at a time."""
        current = list(initial_branches)
        improved = True
        while improved and _time.time() < deadline:
            improved = False
            # Try adding each candidate, prefer those that keep paths short
            candidates = [v for v in range(n) if v not in current]
            # Sort candidates by how well-connected they are to existing branches
            def cand_score(v):
                score = 0
                for u in current:
                    if v in adj[u]:
                        score += 1
                return score
            candidates.sort(key=cand_score, reverse=True)
            for candidate in candidates:
                new_set = current + [candidate]
                paths = check_branch_set(new_set)
                if paths is not None:
                    current = new_set
                    improved = True
                    break
        return current

    def local_search(branches):
        """Try to improve by swapping out a branch for a better one."""
        current = list(branches)
        improved = True
        while improved and _time.time() < deadline:
            improved = False
            for i in range(len(current)):
                for candidate in range(n):
                    if candidate in current:
                        continue
                    new_set = current[:i] + [candidate] + current[i+1:]
                    paths = check_branch_set(new_set)
                    if paths is not None:
                        # Check if this is actually an improvement (same size, but maybe better paths)
                        # For now, just accept any valid swap to escape local optima
                        current = new_set
                        improved = True
                        break
                if improved:
                    break
        return current

    # Find maximum clique using Bron-Kerbosch
    def bron_kerbosch(R, P_set, X):
        """Find all maximal cliques. Returns list of cliques (as sorted tuples)."""
        if not P_set and not X:
            return [tuple(sorted(R))]
        cliques = []
        # Pivot
        if P_set or X:
            pivot_candidates = list(P_set | X)
            pivot = max(pivot_candidates, key=lambda v: len(adj[v] & P_set))
        else:
            return []
        candidates = [v for v in P_set if v not in adj[pivot]]
        for v in candidates:
            new_R = R | {v}
            new_P = P_set & adj[v]
            new_X = X & adj[v]
            cliques.extend(bron_kerbosch(new_R, new_P, new_X))
            P_set.discard(v)
            X.add(v)
        return cliques

    # Find maximum clique
    max_clique = set()
    all_cliques = bron_kerbosch(set(), set(range(n)), set())
    for c in all_cliques:
        if len(c) > len(max_clique):
            max_clique = set(c)

    # Build initial branch sets from multiple strategies
    best_branches = None
    best_paths = None

    def update_best(b, p):
        nonlocal best_branches, best_paths
        if b is not None and p is not None:
            if best_branches is None or len(b) > len(best_branches):
                best_branches = list(b)
                best_paths = p
            elif len(b) == len(best_branches):
                # Prefer the one with fewer total interior vertices
                total_new = sum(len(path[1:-1]) for path in p.values())
                total_old = sum(len(path[1:-1]) for path in best_paths.values())
                if total_new < total_old:
                    best_branches = list(b)
                    best_paths = p

    # Strategy 1: Start from maximum clique
    if len(max_clique) >= 2:
        grown = try_grow(list(max_clique))
        paths = check_branch_set(grown)
        update_best(grown, paths)

    # Strategy 2: Start from top-degree vertices
    deg = [len(adj[i]) for i in range(n)]
    top = sorted(range(n), key=lambda x: -deg[x])[:8]
    grown = try_grow(list(top[:4]))
    paths = check_branch_set(grown)
    update_best(grown, paths)

    grown = try_grow(list(top[:5]))
    paths = check_branch_set(grown)
    update_best(grown, paths)

    # Strategy 3: Start from various subsets of high-degree vertices
    import random
    random.seed(12345)
    for _ in range(30):
        if _time.time() > deadline:
            break
        start_size = random.randint(2, 5)
        start = random.sample(range(n), start_size)
        grown = try_grow(start)
        paths = check_branch_set(grown)
        update_best(grown, paths)

    # Strategy 4: Local search from current best
    if best_branches is not None and _time.time() < deadline:
        improved = local_search(best_branches)
        if len(improved) > len(best_branches):
            paths = check_branch_set(improved)
            update_best(improved, paths)

    # Strategy 5: Try all pairs of high-degree vertices as seeds and grow
    if _time.time() < deadline:
        high_deg = [v for v in range(n) if deg[v] >= 5]
        for i in range(len(high_deg)):
            if _time.time() > deadline:
                break
            for j in range(i + 1, len(high_deg)):
                if _time.time() > deadline:
                    break
                seed = [high_deg[i], high_deg[j]]
                grown = try_grow(seed)
                paths = check_branch_set(grown)
                update_best(grown, paths)

    # Fallback: at least 2 branches
    if best_branches is None or len(best_branches) < 2:
        u, v = edges[0]
        best_branches = [u, v]
        best_paths = {(u, v): [u, v]}

    # Format output
    branches = [int(b) for b in best_branches]
    paths_list = []
    for (u, v) in combinations(branches, 2):
        path = best_paths.get((u, v)) or best_paths.get((v, u))
        if path is None:
            # Should not happen, but find a path as fallback
            path = find_shortest_path(u, v, set(best_branches) - {u, v})
            if path is None:
                path = [u, v]
        paths_list.append([int(x) for x in path])

    return {'branches': branches, 'paths': paths_list}
# EVOLVE-BLOCK-END
