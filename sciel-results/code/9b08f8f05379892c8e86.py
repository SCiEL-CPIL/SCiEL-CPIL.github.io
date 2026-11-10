"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    deadline = time.time() + 80  # safety margin before 90s kill
    n = P['n']
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    def bfs_path(src, dst, forbidden):
        """Find a simple path from src to dst avoiding vertices in forbidden (except src, dst)."""
        if src == dst:
            return None
        if dst in forbidden and dst != src and dst != dst:
            return None
        # forbidden contains vertices that cannot be used as interior
        # src and dst are allowed
        visited = {src}
        parent = {src: None}
        queue = deque([src])
        while queue:
            u = queue.popleft()
            if u == dst:
                # Reconstruct path
                path = []
                cur = dst
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path
            for v in adj[u]:
                if v not in visited:
                    # v can be visited if it's dst or not in forbidden
                    if v == dst or (v not in forbidden):
                        visited.add(v)
                        parent[v] = u
                        queue.append(v)
        return None
    
    def can_route(branches):
        """Try to find disjoint paths for all pairs of branches. Returns dict of paths or None."""
        pairs = list(combinations(branches, 2))
        # We need to assign each pair a path such that all interior vertices are disjoint
        # Strategy: process pairs one by one, each time finding a path avoiding already-used interiors
        used_interiors = set()
        paths = {}
        for (u, v) in pairs:
            # Find path from u to v avoiding used_interiors (interior vertices)
            # u and v are branches, not in used_interiors
            path = bfs_path(u, v, used_interiors)
            if path is None:
                return None
            # Check that path is valid: endpoints are u,v, interiors don't hit any branch
            if path[0] != u or path[-1] != v:
                return None
            for mid in path[1:-1]:
                if mid in branches:
                    return None
                if mid in used_interiors:
                    return None
            used_interiors.update(path[1:-1])
            paths[(u, v)] = path
        return paths
    
    # Find a good initial clique
    def find_clique_of_size(k):
        """Find a clique of size k using simple backtracking."""
        if k <= 1:
            return list(range(k))
        
        # Try all combinations for small k
        for combo in combinations(range(n), k):
            is_clique = True
            for a, b in combinations(combo, 2):
                if b not in adj[a]:
                    is_clique = False
                    break
            if is_clique:
                return list(combo)
        return None
    
    # Start with a clique of size 4, then try to expand
    best_branches = None
    best_paths = None
    
    # Try to find a clique of size 4 first
    clique4 = find_clique_of_size(4)
    if clique4 is None:
        clique4 = find_clique_of_size(3)
    if clique4 is None:
        # Fallback: just use two adjacent vertices
        u, v = P['edges'][0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
    
    # Try to expand the clique greedily
    branches = list(clique4)
    
    # Sort candidate vertices by degree (descending) for greedy addition
    candidates = [v for v in range(n) if v not in branches]
    candidates.sort(key=lambda v: -len(adj[v]))
    
    for cand in candidates:
        if time.time() > deadline - 5:
            break
        new_branches = branches + [cand]
        paths = can_route(new_branches)
        if paths is not None:
            branches = new_branches
            best_branches = list(branches)
            best_paths = {k: list(p) for k, p in paths.items()}
    
    # Also try different initial cliques
    for clique_size in [5, 4, 3]:
        if time.time() > deadline - 5:
            break
        # Try all combinations of size clique_size
        for combo in combinations(range(n), clique_size):
            if time.time() > deadline - 3:
                break
            # Check if it's a clique
            is_clique = True
            for a, b in combinations(combo, 2):
                if b not in adj[a]:
                    is_clique = False
                    break
            if not is_clique:
                continue
            branches = list(combo)
            # Try to expand
            candidates = [v for v in range(n) if v not in branches]
            candidates.sort(key=lambda v: -len(adj[v]))
            for cand in candidates:
                if time.time() > deadline - 3:
                    break
                new_branches = branches + [cand]
                paths = can_route(new_branches)
                if paths is not None:
                    branches = new_branches
                    if best_branches is None or len(branches) > len(best_branches):
                        best_branches = list(branches)
                        best_paths = {k: list(p) for k, p in paths.items()}
    
    if best_branches is None or len(best_branches) < 2:
        u, v = P['edges'][0]
        return {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
    
    # Convert to required format
    result_branches = [int(b) for b in best_branches]
    result_paths = []
    for (u, v), path in best_paths.items():
        result_paths.append([int(x) for x in path])
    
    return {'branches': result_branches, 'paths': result_paths}
# EVOLVE-BLOCK-END
