"""Search for maximum branch count in a pinned graph clique subdivision."""
import time
import random
import sys

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}


def construct():
    t_start = time.time()
    DEADLINE = 80.0  # safety margin under 90s wall-clock

    n = P['n']
    edges = P['edges']

    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    def time_up():
        return (time.time() - t_start) > DEADLINE

    # BFS shortest path avoiding banned interior vertices
    def bfs_path(s, t, banned):
        """Find shortest simple path from s to t avoiding banned in interior."""
        if s == t:
            return None
        # Direct edge
        if t in adj[s] and t not in banned:
            return [s, t]
        from collections import deque
        parent = {s: None}
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v == s:
                    continue
                if v in parent:
                    continue
                if v == t:
                    # Reconstruct
                    path = [v]
                    cur = u
                    while cur is not None:
                        path.append(cur)
                        cur = parent[cur]
                    path.reverse()
                    return path
                if v in banned:
                    continue
                parent[v] = u
                q.append(v)
        return None

    # Greedy: build branch set by adding vertices one at a time
    # For a given ordered list of candidate branches, try to find disjoint paths
    def try_build(branches):
        """Given a set of branches, find internally-disjoint paths for each pair."""
        m = len(branches)
        if m < 2:
            return None
        used_interior = set()
        paths = {}
        # Order pairs: process pairs that are "easier" first (adjacent branches)
        pairs = []
        for i in range(m):
            for j in range(i + 1, m):
                pairs.append((i, j))
        # Sort: adjacent pairs first (direct edge possible), then by some heuristic
        def pair_key(p):
            i, j = p
            a, b = branches[i], branches[j]
            # Prefer pairs with direct edge
            direct = 0 if b in adj[a] else 1
            # Prefer pairs where both have high degree to available vertices
            return (direct, -len(adj[a]), -len(adj[b]))
        pairs.sort(key=pair_key)

        for (i, j) in pairs:
            a, b = branches[i], branches[j]
            # Find path avoiding used_interior and all other branches
            banned = set(used_interior)
            banned.update(branches)
            banned.discard(a)
            banned.discard(b)
            path = bfs_path(a, b, banned)
            if path is None:
                return None
            # Record interior
            interior = set(path[1:-1])
            if interior & used_interior:
                return None
            used_interior.update(interior)
            paths[(i, j)] = path
        # Build result
        result_paths = []
        for (i, j) in pairs:
            result_paths.append(paths[(i, j)])
        return result_paths

    # Strategy: try to find a large clique-like set
    # First, find a maximum clique as a starting point, then try to expand
    def find_max_clique():
        # Simple branch and bound
        best = []
        # Use a greedy coloring bound
        candidates = list(range(n))
        # Order by degree descending
        candidates.sort(key=lambda x: -len(adj[x]))

        def expand(clique, cand):
            nonlocal best
            if len(clique) > len(best):
                best = list(clique)
            if not cand:
                return
            for i, v in enumerate(cand):
                new_clique = clique + [v]
                new_cand = [u for u in cand[i + 1:] if u in adj[v]]
                expand(new_clique, new_cand)

        expand([], candidates)
        return best

    # Get initial clique
    clique = find_max_clique()

    best_branches = None
    best_paths = None
    best_count = 0

    def update_best(branches, paths):
        nonlocal best_branches, best_paths, best_count
        if paths is not None and len(branches) > best_count:
            best_branches = branches
            best_paths = paths
            best_count = len(branches)

    # Try the clique
    if len(clique) >= 2:
        paths = try_build(clique)
        if paths is not None:
            update_best(clique, paths)

    # Try to expand the clique by adding vertices
    # Greedy: try adding each non-clique vertex
    current = list(clique)
    changed = True
    while changed and not time_up():
        changed = False
        for v in range(n):
            if v in current:
                continue
            trial = current + [v]
            paths = try_build(trial)
            if paths is not None:
                current = trial
                changed = True
                update_best(current, paths)
                break
        if time_up():
            break

    # Also try random restarts with different initial seeds
    if not time_up():
        for restart in range(50):
            if time_up():
                break
            # Random subset of size around best_count
            seed_size = min(n, best_count + random.randint(0, 3))
            seed = random.sample(range(n), seed_size)
            # Try to expand
            current = list(seed)
            changed = True
            while changed and not time_up():
                changed = False
                for v in range(n):
                    if v in current:
                        continue
                    trial = current + [v]
                    paths = try_build(trial)
                    if paths is not None:
                        current = trial
                        changed = True
                        update_best(current, paths)
                        break
                if time_up():
                    break

    # Final fallback: at least 2 branches
    if best_branches is None or len(best_branches) < 2:
        u, v = edges[0]
        best_branches = [u, v]
        best_paths = [[u, v]]

    # Ensure types are correct Python int and list
    branches_out = [int(b) for b in best_branches]
    paths_out = [[int(x) for x in p] for p in best_paths]

    return {'branches': branches_out, 'paths': paths_out}
