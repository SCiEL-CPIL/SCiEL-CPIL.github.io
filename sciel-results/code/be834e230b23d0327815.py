"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Find all maximal cliques using Bron-Kerbosch with pivot
    def bron_kerbosch(R, P_set, X, pivot=None):
        if time.time() > deadline:
            return
        if not P_set and not X:
            yield R
            return
        if pivot is None:
            pivot = next(iter(P_set | X)) if (P_set or X) else None
        candidates = P_set - adj.get(pivot, set()) if pivot is not None else P_set
        for v in list(candidates):
            new_R = R | {v}
            new_P = P_set & adj.get(v, set())
            new_X = X & adj.get(v, set())
            yield from bron_kerbosch(new_R, new_P, new_X)
            P_set.remove(v)
            X.add(v)
    
    # Get all maximal cliques
    all_cliques = list(bron_kerbosch(set(), set(range(n)), set()))
    
    # Sort by size descending
    all_cliques.sort(key=len, reverse=True)
    
    # Also generate all subsets of cliques, sorted by size
    # But first, check cliques themselves
    
    def find_path_bfs(u, v, forbidden):
        """Find shortest path from u to v avoiding forbidden vertices in interior."""
        if u == v:
            return None
        if v in forbidden and v != u and v != v:  # v is endpoint, ok
            pass
        # BFS
        parent = {u: None}
        queue = deque([u])
        while queue:
            cur = queue.popleft()
            if cur == v:
                # Reconstruct path
                path = []
                node = v
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return path
            for nb in adj[cur]:
                if nb not in parent:
                    # Interior vertices must not be in forbidden
                    if nb == v or nb == u or nb not in forbidden:
                        parent[nb] = cur
                        queue.append(nb)
        return None
    
    def check_branch_set(branches):
        """Check if branch set is feasible. Returns paths dict or None."""
        if len(branches) < 2:
            return None
        branches_set = set(branches)
        used_interiors = set()
        paths = {}
        
        # Order pairs by some heuristic
        pairs = list(combinations(branches, 2))
        
        for u, v in pairs:
            # Find path from u to v, interior avoids used_interiors and all branches except u,v
            forbidden = used_interiors | (branches_set - {u, v})
            path = find_path_bfs(u, v, forbidden)
            if path is None:
                return None
            # Extract interior
            interior = set(path[1:-1])
            if interior & (branches_set - {u, v}):
                return None
            if interior & used_interiors:
                return None
            used_interiors |= interior
            paths[(u, v)] = path
        
        return paths
    
    # Try to find the best feasible branch set
    best_result = None
    best_size = 1
    
    # First try all cliques from largest to smallest
    for clique in all_cliques:
        if time.time() > deadline:
            break
        if len(clique) <= best_size:
            break
        branches = sorted(clique)
        paths = check_branch_set(branches)
        if paths is not None:
            best_result = (branches, paths)
            best_size = len(branches)
            break  # Found largest clique that works
    
    # If no clique worked, try subsets
    if best_result is None:
        # Try pairs
        for u, v in combinations(range(n), 2):
            if time.time() > deadline:
                break
            if (u, v) not in [(a, b) for a, b in edges] and (v, u) not in [(a, b) for a, b in edges]:
                # Check if there's any path
                path = find_path_bfs(u, v, set())
                if path is not None:
                    best_result = ([u, v], {(u, v): path})
                    best_size = 2
                    break
    
    if best_result is None:
        # Fallback: first edge
        u, v = edges[0]
        best_result = ([u, v], {(u, v): [u, v]})
    
    branches, paths = best_result
    
    # Format output
    branch_list = [int(b) for b in sorted(branches)]
    path_list = []
    for (u, v) in combinations(branch_list, 2):
        if (u, v) in paths:
            p = [int(x) for x in paths[(u, v)]]
        elif (v, u) in paths:
            p = [int(x) for x in paths[(v, u)]]
        else:
            p = [u, v]
        path_list.append(p)
    
    return {'branches': branch_list, 'paths': path_list}
# EVOLVE-BLOCK-END
