"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

DEADLINE = time.time() + 85.0

def construct():
    t0 = time.time()
    n = P['n']
    edges = P['edges']
    edge_set = set()
    for e in edges:
        a, b = e
        edge_set.add((a, b))
        edge_set.add((b, a))
    adj = [set() for _ in range(n)]
    for e in edges:
        a, b = e
        adj[a].add(b)
        adj[b].add(a)
    adj_list = [sorted(adj[i]) for i in range(n)]

    def time_up():
        return time.time() - t0 > DEADLINE - 3.0

    def fallback():
        u, v = edges[0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}

    def bfs_shortest_path(s, t, blocked):
        """BFS shortest path from s to t avoiding blocked vertices (except s, t)."""
        if s == t:
            return [s]
        prev = {s: None}
        q = deque([s])
        while q:
            cur = q.popleft()
            if cur == t:
                break
            for nb in adj_list[cur]:
                if nb in prev:
                    continue
                if nb in blocked and nb != s and nb != t:
                    continue
                prev[nb] = cur
                q.append(nb)
        if t not in prev:
            return None
        path = []
        cur = t
        while cur is not None:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        return path

    def solve_disjoint_paths(branches):
        """
        Try to find internally vertex-disjoint paths for all pairs of branches.
        Uses backtracking with good ordering.
        Returns list of paths or None.
        """
        bs = set(branches)
        pairs = list(combinations(sorted(branches), 2))
        
        # Order pairs by difficulty: longer shortest path first (more constrained)
        def pair_difficulty(pair):
            u, v = pair
            blocked = bs - {u, v}
            path = bfs_shortest_path(u, v, blocked)
            if path is None:
                return -1  # infeasible
            return -len(path)  # longer path = harder, sort first
        
        scored = []
        for p in pairs:
            d = pair_difficulty(p)
            if d == -1:
                return None
            scored.append((d, p))
        scored.sort()
        ordered_pairs = [p for (_, p) in scored]
        
        # Backtracking search
        used = set()
        paths_found = []
        
        def backtrack(idx):
            if time_up():
                return False
            if idx == len(ordered_pairs):
                return True
            u, v = ordered_pairs[idx]
            blocked = used | (bs - {u, v})
            # Find a path from u to v avoiding blocked
            path = bfs_shortest_path(u, v, blocked)
            if path is None:
                return False
            interior = set(path[1:-1])
            if interior & used:
                return False
            used.add(interior) if False else used.update(interior)
            paths_found.append(path)
            if backtrack(idx + 1):
                return True
            paths_found.pop()
            used.difference_update(interior)
            return False
        
        if backtrack(0):
            return paths_found
        return None

    def try_branch_set(branches):
        """Try to find disjoint paths for the given branch set."""
        if len(branches) < 2:
            return None
        return solve_disjoint_paths(branches)

    best = None
    best_size = 0

    # Start with 2-branch
    u, v = edges[0]
    paths = try_branch_set([u, v])
    if paths is not None:
        best = {'branches': [int(u), int(v)], 'paths': [[int(x) for x in pp] for pp in paths]}
        best_size = 2

    vertices = list(range(n))
    
    # Try to find larger branch sets systematically
    # Start from size 3 up, but also try large sizes first for early pruning
    # Strategy: try all combinations of size k for k from best_size+1 upward
    # But that's too expensive. Instead, use a greedy + local search approach.
    
    # First: try all triangles (size 3)
    triangles = []
    for tri in combinations(vertices, 3):
        a, b, c = tri
        if (a, b) in edge_set and (b, c) in edge_set and (a, c) in edge_set:
            triangles.append(tri)
    
    for tri in triangles:
        if time_up():
            break
        paths = try_branch_set(list(tri))
        if paths is not None and best_size < 3:
            best = {'branches': [int(x) for x in tri], 'paths': [[int(x) for x in pp] for pp in paths]}
            best_size = 3

    # Greedy expansion: add one vertex at a time
    if best_size >= 2:
        base = sorted(best['branches'])
    else:
        base = sorted([u, v])
    
    improved = True
    max_rounds = 15
    for _ in range(max_rounds):
        if not improved or time_up():
            break
        improved = False
        current = sorted(base)
        non_branch = [x for x in vertices if x not in set(current)]
        # Sort by degree descending
        non_branch.sort(key=lambda x: len(adj[x]), reverse=True)
        for vb in non_branch:
            if time_up():
                break
            new_bs = current + [vb]
            paths = try_branch_set(new_bs)
            if paths is not None:
                base = sorted(new_bs)
                best = {'branches': [int(x) for x in base], 'paths': [[int(x) for x in pp] for pp in paths]}
                best_size = len(base)
                improved = True
                break

    # Try to expand further from multiple starting points
    if not time_up() and best_size < 10:
        # Try expanding from different initial sets
        for start_size in range(3, min(6, best_size + 1)):
            if time_up():
                break
            # Try random subsets of this size as starting points
            import random
            random.seed(42)
            for trial in range(50):
                if time_up():
                    break
                subset = random.sample(vertices, start_size)
                paths = try_branch_set(subset)
                if paths is not None:
                    current = sorted(subset)
                    # Expand
                    for _ in range(10):
                        if time_up():
                            break
                        non_branch = [x for x in vertices if x not in set(current)]
                        non_branch.sort(key=lambda x: len(adj[x]), reverse=True)
                        added = False
                        for vb in non_branch:
                            if time_up():
                                break
                            new_bs = current + [vb]
                            paths2 = try_branch_set(new_bs)
                            if paths2 is not None:
                                current = sorted(new_bs)
                                if len(current) > best_size:
                                    best = {'branches': [int(x) for x in current], 'paths': [[int(x) for x in pp] for pp in paths2]}
                                    best_size = len(current)
                                added = True
                                break
                        if not added:
                            break

    # Direct search for large branch sets
    if not time_up() and best_size < 10:
        # Try all 4-combinations
        for quad in combinations(vertices, 4):
            if time_up():
                break
            if best_size >= 4:
                break
            paths = try_branch_set(list(quad))
            if paths is not None:
                best = {'branches': [int(x) for x in quad], 'paths': [[int(x) for x in pp] for pp in paths]}
                best_size = 4

    if best is None:
        return fallback()

    return best
# EVOLVE-BLOCK-END
