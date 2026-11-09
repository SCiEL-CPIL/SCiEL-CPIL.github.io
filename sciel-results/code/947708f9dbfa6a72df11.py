"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
from collections import deque
import sys

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Precompute edge set for quick lookup
    edge_set = set()
    for u, v in edges:
        edge_set.add((u, v))
        edge_set.add((v, u))
    
    def find_paths(branches):
        """
        Try to find pairwise internally disjoint paths between all pairs of branches.
        Returns dict mapping (i,j) to path, or None if not possible.
        """
        k = len(branches)
        if k < 2:
            return None
        
        # Map branch indices to actual vertices
        branch_to_idx = {v: i for i, v in enumerate(branches)}
        
        # Non-branch vertices
        non_branches = [v for v in range(n) if v not in branch_to_idx]
        
        # We'll try to find paths using a greedy approach with backtracking
        # For each pair, find a shortest path avoiding already-used interior vertices
        
        used_interior = set()
        paths = {}
        
        # Sort pairs by some heuristic - maybe by distance
        pairs = list(combinations(range(k), 2))
        
        # Try to order pairs by shortest path length (greedy)
        def bfs_shortest_path(src, dst, forbidden):
            """Find shortest path from src to dst avoiding forbidden vertices (except src, dst)."""
            if src == dst:
                return [src]
            visited = {src}
            queue = deque([(src, [src])])
            while queue:
                curr, path = queue.popleft()
                for nb in adj[curr]:
                    if nb == dst:
                        return path + [dst]
                    if nb in forbidden or nb in visited:
                        continue
                    visited.add(nb)
                    queue.append((nb, path + [nb]))
            return None
        
        # Sort pairs by initial shortest path length
        pair_lengths = []
        for i, j in pairs:
            sp = bfs_shortest_path(branches[i], branches[j], set())
            if sp is None:
                return None
            pair_lengths.append((len(sp), i, j))
        
        pair_lengths.sort()
        
        # Greedily assign paths
        for _, i, j in pair_lengths:
            sp = bfs_shortest_path(branches[i], branches[j], used_interior)
            if sp is None:
                return None
            # Add interior vertices to used set
            for v in sp[1:-1]:
                if v in branch_to_idx:
                    return None  # Path goes through another branch
                used_interior.add(v)
            paths[(i, j)] = sp
        
        return paths
    
    def try_subset(branches):
        """Try to find valid paths for the given subset of branches."""
        paths = find_paths(branches)
        if paths is None:
            return None
        # Convert to output format
        result_paths = []
        for (i, j), path in paths.items():
            result_paths.append(list(path))
        return {'branches': list(branches), 'paths': result_paths}
    
    # Strategy: Start with a large candidate set and try to find a valid clique subdivision
    # First, try to find the largest possible set
    
    # Start with all vertices
    all_vertices = list(range(n))
    
    # Try decreasing sizes
    best_result = None
    
    # First check if the whole graph works
    result = try_subset(all_vertices)
    if result is not None:
        return result
    
    # Try subsets of size n-1, n-2, etc.
    # But this is exponential. Instead, use a greedy approach:
    # Start with a seed set and try to add vertices one by one
    
    # Find a good initial set: use vertices with high degree
    degrees = [len(adj[v]) for v in range(n)]
    sorted_vertices = sorted(range(n), key=lambda v: -degrees[v])
    
    # Try building up from high-degree vertices
    current_set = []
    for v in sorted_vertices:
        trial_set = current_set + [v]
        if len(trial_set) >= 2:
            result = try_subset(trial_set)
            if result is not None:
                current_set = trial_set
                best_result = result
    
    if best_result is not None:
        return best_result
    
    # Fallback: find any triangle
    for row in combinations(range(n), 3):
        if all((row[i], row[j]) in edge_set for i in range(3) for j in range(i+1, 3)):
            return {'branches': list(row), 'paths': [[row[i], row[j]] for i in range(3) for j in range(i+1, 3)]}
    
    # Fallback: find any edge
    if edges:
        u, v = edges[0]
        return {'branches': [u, v], 'paths': [[u, v]]}
    
    # Should not reach here
    return {'branches': [0, 1], 'paths': [[0, 1]]}

# EVOLVE-BLOCK-END
