"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    adj = [[] for _ in range(n)]
    adj_set = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        adj_set[u].add(v)
        adj_set[v].add(u)
    
    # 1. Find Max Clique (exact for small n)
    # Bron-Kerbosch with pivot
    def bronk(R, P, X):
        if not P and not X:
            return R
        max_r = None
        # Simple pivot: pick vertex from P+X with max degree in P
        if P or X:
            pivot = max(P.union(X), key=lambda v: len(adj_set[v].intersection(P)))
            candidates = [v for v in P if v not in adj_set[pivot]]
        else:
            candidates = list(P)
            
        for v in candidates:
            R_new = R.union([v])
            P_new = P.intersection(adj_set[v])
            X_new = X.intersection(adj_set[v])
            res = bronk(R_new, P_new, X_new)
            if res is not None:
                if max_r is None or len(res) > len(max_r):
                    max_r = res
            P.remove(v)
            X.add(v)
        return max_r

    max_clique = bronk(set(), set(range(n)), set())
    if max_clique and len(max_clique) >= 4:
        branches = sorted(list(max_clique))
        paths = []
        for u, v in combinations(branches, 2):
            paths.append([u, v])
        return {'branches': branches, 'paths': paths}

    # 2. If no K4 clique, search for K4 subdivision
    # We will try to find 4 branches and disjoint paths.
    # To be efficient, we only check a subset of 4-tuples or use a fast heuristic.
    
    # Precompute all-pairs shortest paths for heuristic sorting
    # BFS from each node
    dist = [[-1]*n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
        q = deque([i])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[i][v] == -1:
                    dist[i][v] = dist[i][u] + 1
                    q.append(v)
    
    # Generate 4-tuples and sort by sum of pairwise distances (shorter is better for disjointness)
    candidates = []
    for combo in combinations(range(n), 4):
        d_sum = 0
        for u, v in combinations(combo, 2):
            if dist[combo[0]][v] == -1: # disconnected
                d_sum = float('inf')
                break
            d_sum += dist[u][v]
        candidates.append((d_sum, combo))
        
    candidates.sort()
    
    start_time = time.time()
    deadline = start_time + 45.0 # 45s limit
    
    def find_disjoint_paths(branches):
        # branches: list of 4 vertices
        # pairs: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
        pairs = list(combinations(branches, 2))
        # We need to find a path for each pair.
        # Path is a list of vertices.
        # Constraints:
        # 1. Endpoints are the branch vertices.
        # 2. Internal vertices are distinct across all paths.
        # 3. Internal vertices are not branch vertices.
        
        # We will use backtracking.
        # Order pairs by "difficulty" (longest shortest path first? or just fixed order)
        # Let's try fixed order first.
        
        used_internal = set()
        
        def dfs(idx, current_paths):
            if time.time() > deadline:
                return None
            if idx == len(pairs):
                return current_paths
            
            u, v = pairs[idx]
            # Find a path from u to v avoiding used_internal and branches (except endpoints)
            # BFS to find shortest path first
            # If shortest path conflicts, try next shortest?
            # For simplicity, let's just try to find ANY path that doesn't conflict.
            # We can use BFS with state (node, visited_internal_set) but that's expensive.
            # Instead, we can iterate over possible intermediate vertices.
            
            # Simple heuristic: Try shortest path first.
            # Get all shortest paths? No, just one.
            # If it fails, we need to backtrack.
            
            # To make this work, we need a way to find an alternative path.
            # Let's use a DFS that explores neighbors, avoiding used_internal and branches.
            
            # Optimization: If u and v are adjacent, path is [u, v].
            if v in adj_set[u]:
                return current_paths + [[u, v]]
                
            # BFS to find a path
            # We need to ensure the path's internal vertices are not in used_internal.
            # And not in branches.
            
            # Let's do a BFS that records parent, but we need to check conflicts.
            # Actually, a simple DFS with backtracking on the path construction is better.
            
            # Let's try to find a path using BFS, but if we hit a conflict, we stop.
            # This is not complete.
            # For n=24, we can afford a more thorough search.
            
            # Let's use a recursive search for the path from u to v.
            # State: current_node, visited_in_path
            # Constraint: current_node not in used_internal, not in branches (unless endpoint).
            
            def find_path(src, dst, visited):
                # visited is a set of nodes already in the path (including src)
                if src == dst:
                    return [src]
                for nb in adj[src]:
                    if nb in visited:
                        continue
                    if nb in used_internal:
                        continue
                    if nb in branches and nb != dst:
                        continue
                    # Allow nb if it is dst
                    if nb == dst:
                        return [src, dst]
                    # Recurse
                    visited.add(nb)
                    res = find_path(nb, dst, visited)
                    if res:
                        return [src] + res
                    visited.remove(nb)
                return None
            
            # Try to find a path
            path = find_path(u, v, {u})
            if path:
                # Check if internal vertices are valid
                internals = set(path[1:-1])
                if not (internals & used_internal) and not (internals & set(branches)):
                    used_internal.update(internals)
                    res = dfs(idx + 1, current_paths + [path])
                    if res:
                        return res
                    used_internal.difference_update(internals)
            return None
            
        return dfs(0, [])

    for d_sum, combo in candidates:
        if time.time() > deadline:
            break
        if d_sum == float('inf'):
            continue
        # Heuristic: skip if sum of distances is too large?
        # For K4 subdivision, we need enough "room".
        # Let's just try.
        
        # To speed up, we can skip if any pair is disconnected (already checked)
        # Also, if the 4 vertices are not in the same biconnected component?
        # Let's just run the search.
        
        res = find_disjoint_paths(list(combo))
        if res:
            return {'branches': list(combo), 'paths': res}

    # Fallback: Return best K3 (triangle)
    # Find a triangle
    for u, v, w in combinations(range(n), 3):
        if v in adj_set[u] and w in adj_set[v] and u in adj_set[w]:
            return {'branches': [u, v, w], 'paths': [[u, v], [u, w], [v, w]]}
            
    # Fallback: Edge
    u, v = edges[0]
    return {'branches': [u, v], 'paths': [[u, v]]}

# EVOLVE-BLOCK-END
