import time
import random
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    t_start = time.time()
    DEADLINE = 85.0

    n = P['n']
    edges = P['edges']
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    adj_set = adj
    degrees = [len(adj[v]) for v in range(n)]
    sorted_verts = sorted(range(n), key=lambda x: degrees[x], reverse=True)

    def time_up():
        return time.time() - t_start > DEADLINE

    def find_shortest_path(u, v, banned):
        """Find shortest path from u to v avoiding banned vertices (interior)."""
        if u == v:
            return None
        if v in adj_set[u]:
            return [u, v]
        parent = {u: None}
        q = deque([u])
        while q:
            cur = q.popleft()
            for nxt in adj_set[cur]:
                if nxt == v:
                    path = [v]
                    node = cur
                    while node is not None:
                        path.append(node)
                        node = parent[node]
                    path.reverse()
                    return path
                if nxt in parent or nxt in banned:
                    continue
                parent[nxt] = cur
                q.append(nxt)
        return None

    def find_paths_for_branches(branches, used_interior):
        """
        Try to find vertex-disjoint paths for all pairs of branches.
        Uses a greedy approach with backtracking on ordering.
        Returns dict of frozenset->path or None if fails.
        """
        bset = set(branches)
        pairs = list(combinations(branches, 2))
        
        # Try multiple orderings
        orderings = []
        
        # Order 1: by sum of degrees (ascending - harder pairs first)
        orderings.append(sorted(pairs, key=lambda p: degrees[p[0]] + degrees[p[1]]))
        # Order 2: reverse
        orderings.append(list(reversed(orderings[0])))
        # Order 3: by number of common neighbors (ascending)
        orderings.append(sorted(pairs, key=lambda p: len(adj_set[p[0]] & adj_set[p[1]])))
        # Order 4: reverse
        orderings.append(list(reversed(orderings[2])))
        # Order 5: by minimum degree in pair
        orderings.append(sorted(pairs, key=lambda p: min(degrees[p[0]], degrees[p[1]])))
        # Order 6: random
        import random as _r
        for _ in range(5):
            order = list(pairs)
            _r.shuffle(order)
            orderings.append(order)
        
        for order in orderings:
            if time_up():
                break
            paths = {}
            ui = set(used_interior)
            success = True
            for a, b in order:
                banned = ui | bset
                banned.discard(a)
                banned.discard(b)
                path = find_shortest_path(a, b, banned)
                if path is None:
                    success = False
                    break
                interior = set(path[1:-1])
                if interior & bset or interior & ui:
                    success = False
                    break
                ui = ui | interior
                paths[frozenset((a, b))] = path
            if success:
                return paths
        return None

    def validate(branches, paths):
        """Validate the subdivision. Returns count if valid, 0 otherwise."""
        if not branches or len(branches) < 2:
            return 0
        bset = set(branches)
        if len(bset) != len(branches):
            return 0
        expected_pairs = set(frozenset((a, b)) for a, b in combinations(branches, 2))
        if len(paths) != len(expected_pairs):
            return 0
        used_interior = set()
        for fkey, path in paths.items():
            if len(path) < 2:
                return 0
            a, b = path[0], path[-1]
            if a not in bset or b not in bset:
                return 0
            if a == b:
                return 0
            if fkey not in expected_pairs:
                return 0
            interior = set(path[1:-1])
            if interior & bset:
                return 0
            if interior & used_interior:
                return 0
            for i in range(len(path) - 1):
                if path[i + 1] not in adj_set[path[i]]:
                    return 0
            used_interior |= interior
        return len(branches)

    best_count = 0
    best_result = None
    rng = random.Random(42)

    # Phase 1: Try to find large clique subdivisions
    # Start with cliques in the graph, then try to expand
    # First, find all maximal cliques
    def find_maximal_cliques():
        """Find all maximal cliques using Bron-Kerbosch."""
        cliques = []
        
        def bron_kerbosch(r, p, x):
            if not p and not x:
                if len(r) >= 3:
                    cliques.append(list(r))
                return
            # Choose pivot to maximize branching
            pivot = max(list(p | x), key=lambda v: len(adj_set[v] & p)) if (p | x) else None
            candidates = list(p - adj_set[pivot]) if pivot is not None else list(p)
            for v in candidates:
                bron_kerbosch(r | {v}, p & adj_set[v], x & adj_set[v])
                p.remove(v)
                x.add(v)
        
        bron_kerbosch(set(), set(range(n)), set())
        return cliques
    
    maximal_cliques = find_maximal_cliques()
    # Sort by size descending
    maximal_cliques.sort(key=lambda c: len(c), reverse=True)
    
    # Try each maximal clique and try to expand it
    for clique in maximal_cliques:
        if time_up():
            break
        if len(clique) <= best_count:
            continue
            
        # Try the clique itself
        paths = find_paths_for_branches(clique, set())
        if paths is not None:
            cnt = validate(clique, paths)
            if cnt > best_count:
                best_count = cnt
                best_result = (list(clique), paths)
        
        # Try to expand: add one vertex at a time
        if not time_up():
            current = list(clique)
            current_paths = paths
            for _ in range(5):  # Try to add up to 5 more vertices
                if time_up():
                    break
                if current_paths is None:
                    break
                added = False
                candidates_to_add = [v for v in range(n) if v not in set(current)]
                # Sort by degree descending
                candidates_to_add.sort(key=lambda v: degrees[v], reverse=True)
                
                for v in candidates_to_add:
                    if time_up():
                        break
                    new_branches = current + [v]
                    new_paths = find_paths_for_branches(new_branches, set())
                    if new_paths is not None:
                        cnt = validate(new_branches, new_paths)
                        if cnt > best_count:
                            best_count = cnt
                            best_result = (new_branches, new_paths)
                            current = new_branches
                            current_paths = new_paths
                            added = True
                            break
                if not added:
                    break
    
    # Phase 2: If we haven't found a good result, try random subsets
    if best_count < 4:
        for k in range(8, 2, -1):
            if time_up():
                break
            for _ in range(50):
                if time_up():
                    break
                cand = rng.sample(range(n), k)
                paths = find_paths_for_branches(cand, set())
                if paths is not None:
                    cnt = validate(cand, paths)
                    if cnt > best_count:
                        best_count = cnt
                        best_result = (cand, paths)
                        break
    
    # Fallback: try all pairs if nothing found
    if best_result is None:
        for u, v in edges:
            branches = [u, v]
            paths = {frozenset((u, v)): [u, v]}
            if validate(branches, paths) > 0:
                best_result = (branches, paths)
                break
        if best_result is None:
            u, v = edges[0]
            branches = [u, v]
            paths = {frozenset((u, v)): [u, v]}
            best_result = (branches, paths)

    branches, paths = best_result

    branch_list = [int(b) for b in sorted(branches)]
    path_list = []
    for a, b in combinations(sorted(branches), 2):
        fkey = frozenset((a, b))
        p = paths.get(fkey)
        if p is None:
            p = [a, b]
        path_list.append([int(x) for x in p])

    return {'branches': branch_list, 'paths': path_list}
