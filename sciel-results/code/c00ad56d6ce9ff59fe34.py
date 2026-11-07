"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50 second safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Precompute all edges as frozensets for quick lookup
    edge_set = set()
    for u, v in edges:
        edge_set.add(frozenset((u, v)))
    
    def has_edge(u, v):
        return frozenset((u, v)) in edge_set
    
    def bfs_path(src, dst, forbidden):
        """Find shortest path from src to dst avoiding forbidden interior vertices."""
        if src == dst:
            return None
        visited = {src}
        parent = {src: None}
        queue = deque([src])
        while queue:
            node = queue.popleft()
            if node == dst:
                # Reconstruct path
                path = []
                cur = dst
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path
            for nb in adj[node]:
                if nb not in visited and (nb == dst or nb not in forbidden):
                    visited.add(nb)
                    parent[nb] = node
                    queue.append(nb)
        return None
    
    def try_assign_paths(branches):
        """Try to assign internally disjoint paths for all pairs of branches.
        Returns dict of (u,v) -> path, or None if fails."""
        if len(branches) < 2:
            return None
        
        branches_set = set(branches)
        paths = {}
        used_interiors = set()
        
        # Sort pairs to process in some order
        pairs = list(combinations(branches, 2))
        
        for u, v in pairs:
            path = bfs_path(u, v, used_interiors)
            if path is None:
                return None
            # Check interior vertices are not branches and not used
            interior = path[1:-1]
            for x in interior:
                if x in branches_set or x in used_interiors:
                    return None
            paths[frozenset((u, v))] = path
            used_interiors.update(interior)
        
        return paths
    
    def greedy_grow(initial_branches):
        """Greedily grow the branch set by adding vertices."""
        current = list(initial_branches)
        current_set = set(current)
        
        # Try to add vertices one by one
        changed = True
        while changed:
            changed = False
            candidates = [v for v in range(n) if v not in current_set]
            # Sort by degree to current branches (prefer high connectivity)
            candidates.sort(key=lambda v: -sum(1 for b in current if has_edge(b, v)))
            
            for cand in candidates:
                if time.time() > deadline:
                    break
                # Check cand connects to all current branches
                if all(has_edge(cand, b) for b in current):
                    new_branches = current + [cand]
                    paths = try_assign_paths(new_branches)
                    if paths is not None:
                        current = new_branches
                        current_set.add(cand)
                        changed = True
                        break
        
        return current, try_assign_paths(current)
    
    # Strategy 1: Start from each edge and greedily grow
    best_result = None
    best_size = 1
    
    for u, v in edges:
        if time.time() > deadline:
            break
        branches, paths = greedy_grow([u, v])
        if paths is not None and len(branches) > best_size:
            best_size = len(branches)
            best_result = (branches, paths)
    
    # Strategy 2: Start from triangles
    for u, v, w in combinations(range(n), 3):
        if time.time() > deadline:
            break
        if has_edge(u, v) and has_edge(v, w) and has_edge(u, w):
            branches, paths = greedy_grow([u, v, w])
            if paths is not None and len(branches) > best_size:
                best_size = len(branches)
                best_result = (branches, paths)
    
    # Fallback
    if best_result is None:
        u, v = edges[0]
        return {'branches': [u, v], 'paths': [[u, v]]}
    
    branches, paths = best_result
    
    # Convert to required format
    branch_list = [int(b) for b in branches]
    path_list = []
    for u, v in combinations(branch_list, 2):
        key = frozenset((u, v))
        p = paths[key]
        path_list.append([int(x) for x in p])
    
    return {'branches': branch_list, 'paths': path_list}

# EVOLVE-BLOCK-END
