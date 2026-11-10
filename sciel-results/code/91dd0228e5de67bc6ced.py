"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import sys
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Helper: check if a set of branches is realizable via max-flow
    def is_realizable(branches):
        if len(branches) < 2:
            return False
        
        # For each pair of branches, we need a path with disjoint interiors
        # We model this as: for each pair (u,v) in branches, find a path
        # such that no interior vertex is used by any other path.
        
        # Strategy: iteratively add pairs, using max-flow to find disjoint paths
        # We'll use a different approach: for each pair, find a shortest path
        # that avoids all previously used interior vertices.
        
        used_interior = set()
        paths_found = []
        
        # Sort pairs by some heuristic (e.g., shortest path length)
        branch_set = set(branches)
        pairs = list(itertools.combinations(branches, 2))
        
        # For each pair, find a path avoiding used_interior
        for u, v in pairs:
            # BFS from u to v avoiding used_interior
            # But we need to ensure the path doesn't go through other branches either
            # Actually, the constraint is: interiors must avoid EVERY branch.
            # So interior vertices cannot be in branch_set at all.
            
            # BFS from u to v, avoiding:
            # - all vertices in branch_set (except u and v themselves)
            # - all vertices in used_interior
            
            forbidden = (branch_set - {u, v}) | used_interior
            
            # BFS
            parent = [-1] * n
            parent[u] = u
            queue = deque([u])
            found = False
            
            while queue:
                cur = queue.popleft()
                if cur == v:
                    found = True
                    break
                for nb in adj[cur]:
                    if parent[nb] == -1 and nb not in forbidden:
                        parent[nb] = cur
                        queue.append(nb)
            
            if not found:
                return None  # Cannot realize this branch set
            
            # Reconstruct path
            path = []
            cur = v
            while cur != u:
                path.append(cur)
                cur = parent[cur]
            path.append(u)
            path.reverse()
            
            # Add interior vertices to used_interior
            for vertex in path[1:-1]:
                used_interior.add(vertex)
            
            paths_found.append(path)
        
        return paths_found
    
    # Strategy: greedily build a large branch set
    # Start with all vertices, then try to remove the "least useful" ones
    # until the set is realizable.
    
    # Better: start with a clique (guaranteed realizable), then try to add vertices
    # that can be connected via longer paths.
    
    # First, find a large clique using Bron-Kerbosch
    max_clique = []
    
    def bron_kerbosch(R, P, X):
        nonlocal max_clique
        if time.time() > deadline:
            return
        
        if len(P) == 0 and len(X) == 0:
            if len(R) > len(max_clique):
                max_clique = list(R)
            return
        
        if len(P) == 0:
            if len(R) > len(max_clique):
                max_clique = list(R)
            return
        
        # Pivot selection
        union_p_x = P.union(X)
        if not union_p_x:
            return
        
        best_pivot = None
        max_uncovered = -1
        for u in union_p_x:
            uncovered = len(P.difference(adj[u]))
            if uncovered > max_uncovered:
                max_uncovered = uncovered
                best_pivot = u
        
        if best_pivot is None:
            return
        
        for v in list(P.difference(adj[best_pivot])):
            bron_kerbosch(R.union([v]), P.intersection(adj[v]), X.intersection(adj[v]))
            P.remove(v)
            X.add(v)
    
    all_vertices = set(range(n))
    bron_kerbosch(set(), all_vertices, set())
    
    if not max_clique:
        max_clique = [0, 1]
    
    # Now try to expand the clique by adding non-clique vertices
    # that can be connected via disjoint paths.
    
    current_branches = list(max_clique)
    
    # Try to add vertices one by one
    candidates = [v for v in range(n) if v not in set(current_branches)]
    
    # Sort candidates by degree (higher degree = more likely to be connectable)
    candidates.sort(key=lambda v: len(adj[v]), reverse=True)
    
    for candidate in candidates:
        if time.time() > deadline:
            break
        
        # Try adding this candidate
        new_branches = current_branches + [candidate]
        paths = is_realizable(new_branches)
        
        if paths is not None:
            current_branches = new_branches
    
    # Final check
    paths = is_realizable(current_branches)
    
    if paths is None:
        # Fallback to the clique
        current_branches = max_clique
        paths = is_realizable(current_branches)
        
        if paths is None:
            # Even the clique failed? Shouldn't happen.
            # Use direct edges for the clique.
            current_branches = max_clique
            paths = []
            for u, v in itertools.combinations(current_branches, 2):
                paths.append([u, v])
    
    # Ensure types are correct
    branches = [int(v) for v in current_branches]
    paths_out = []
    for path in paths:
        paths_out.append([int(v) for v in path])
    
    return {'branches': branches, 'paths': paths_out}

# EVOLVE-BLOCK-END
