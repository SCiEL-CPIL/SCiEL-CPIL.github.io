import time
import random
from itertools import combinations
from collections import deque
import numpy as np

try:
    from scipy.optimize import milp, LinearConstraint, Bounds
    HAS_MILP = True
except ImportError:
    HAS_MILP = False

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

    def validate(branches, paths):
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

    def find_paths_greedy(branches):
        """Find vertex-disjoint paths for given branches using greedy BFS with multiple strategies."""
        bset = set(branches)
        pairs = list(combinations(sorted(branches), 2))
        
        # Precompute shortest path distances between all branch pairs
        # to use as a heuristic for ordering
        def path_length(a, b, banned):
            path = find_shortest_path(a, b, banned)
            if path is None:
                return float('inf')
            return len(path)
        
        # Try multiple orderings
        orderings = []
        
        # Order by sum of degrees (high degree first - more options)
        orderings.append(sorted(pairs, key=lambda p: degrees[p[0]] + degrees[p[1]]))
        orderings.append(list(reversed(orderings[0])))
        
        # Order by common neighbors (fewer common neighbors = less conflict)
        orderings.append(sorted(pairs, key=lambda p: len(adj_set[p[0]] & adj_set[p[1]])))
        orderings.append(list(reversed(orderings[2])))
        
        # Order by shortest path length (shorter paths first - less resource usage)
        def sp_key(p):
            return path_length(p[0], p[1], bset)
        try:
            orderings.append(sorted(pairs, key=sp_key))
            orderings.append(list(reversed(orderings[-1])))
        except:
            pass
        
        # Random orderings
        rng = random.Random(42)
        for _ in range(20):
            order = list(pairs)
            rng.shuffle(order)
            orderings.append(order)
        
        for order in orderings:
            if time_up():
                break
            paths = {}
            ui = set()  # used interior vertices
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

    def try_branch_set(branches):
        """Try to find valid paths for a given branch set. Returns (count, paths) or (0, None)."""
        if len(branches) < 2:
            return 0, None
        paths = find_paths_greedy(branches)
        if paths:
            cnt = validate(branches, paths)
            if cnt > 0:
                return cnt, paths
        return 0, None

    best_count = 0
    best_result = None

    # Phase 1: Try various initial sets with greedy path finding
    initial_sets = [
        sorted_verts[:12],
        sorted_verts[:11],
        sorted_verts[:10],
        sorted_verts[:9],
        sorted_verts[:8],
        sorted_verts[:7],
        [0, 1, 3, 5, 20, 22, 12, 4, 9, 11, 15, 16],
        [0, 1, 3, 5, 20, 22, 12, 4, 9, 11, 15],
        [0, 1, 3, 5, 20, 22, 12, 4, 9, 11],
        [0, 1, 3, 5, 20, 22, 12, 4, 9],
        [0, 1, 3, 5, 9, 20, 22, 12],
        [0, 1, 3, 5, 20, 22, 12, 4],
    ]
    
    for init in initial_sets:
        if time_up():
            break
        if len(init) <= best_count:
            continue
        cnt, paths = try_branch_set(init)
        if cnt > best_count:
            best_count = cnt
            best_result = (list(init), paths)

    # Phase 2: Greedy growth from best found
    if best_result:
        current = list(best_result[0])
        for _ in range(30):
            if time_up():
                break
            added = False
            candidates = [v for v in range(n) if v not in set(current)]
            candidates.sort(key=lambda v: degrees[v], reverse=True)
            for v in candidates:
                new_branches = current + [v]
                cnt, new_paths = try_branch_set(new_branches)
                if cnt > best_count:
                    best_count = cnt
                    best_result = (new_branches, new_paths)
                    current = new_branches
                    added = True
                    break
            if not added:
                break

    # Phase 3: Local search - try swapping out one branch for another
    if best_result and time.time() - t_start < DEADLINE - 5:
        current = list(best_result[0])
        improved = True
        while improved and not time_up():
            improved = False
            cur_set = set(current)
            for i in range(len(current)):
                if time_up():
                    break
                for v in range(n):
                    if v in cur_set:
                        continue
                    new_branches = current[:i] + current[i+1:] + [v]
                    cnt, new_paths = try_branch_set(new_branches)
                    if cnt > best_count:
                        best_count = cnt
                        best_result = (new_branches, new_paths)
                        current = new_branches
                        improved = True
                        break
                if improved:
                    break

    if best_result is None:
        branches = [0, 1]
        paths = {frozenset((0, 1)): [0, 1]}
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
