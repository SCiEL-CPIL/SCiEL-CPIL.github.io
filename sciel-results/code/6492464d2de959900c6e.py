"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    deadline = time.time() + 85
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Find maximum clique using Bron-Kerbosch with pivot
    def find_max_clique(vertices, current_clique):
        if not vertices:
            return current_clique
        # Pick pivot
        pivot = max(vertices, key=lambda v: len(adj[v] & vertices))
        candidates = [v for v in vertices if v not in adj[pivot]]
        if not candidates:
            return current_clique
        
        best = current_clique
        for v in candidates:
            new_clique = current_clique + [v]
            new_vertices = vertices & adj[v]
            result = find_max_clique(new_vertices, new_clique)
            if len(result) > len(best):
                best = result
            if time.time() > deadline:
                break
            vertices.remove(v)
        return best
    
    max_clique = find_max_clique(set(range(n)), [])
    max_k = len(max_clique)
    
    # Function to check if a set of branch vertices can support disjoint paths
    def can_subdivide(branches):
        k = len(branches)
        if k < 2:
            return False
        if k == 2:
            # Just need a path between the two
            return branches[0] in adj[branches[1]]
        
        branch_set = set(branches)
        internal = [v for v in range(n) if v not in branch_set]
        
        # We need k*(k-1)/2 paths, one for each pair
        # Use vertex splitting: each internal vertex has capacity 1
        # Build flow network
        # Source connects to each branch, each branch connects to sink
        # But we need paths between ALL pairs, not just from source
        
        # Alternative: For each pair of branches, find a shortest path
        # Then check if all paths are internally disjoint
        # This is simpler but may not find optimal paths
        
        # Let's try: find shortest path for each pair, check disjointness
        # If not disjoint, try BFS with constraints
        
        # First, try simple approach: shortest paths
        paths = {}
        for i, j in combinations(range(k), 2):
            u, v = branches[i], branches[j]
            # BFS for shortest path avoiding internal vertices already used
            # Actually, let's just find any path first
            path = bfs_path(u, v, adj, branch_set)
            if path is None:
                return False
            paths[(i,j)] = path
        
        # Check if all paths are internally disjoint
        used_internal = set()
        for (i,j), path in paths.items():
            for vertex in path[1:-1]:  # interior vertices
                if vertex in branch_set:
                    return False
                if vertex in used_internal:
                    return False
                used_internal.add(vertex)
        
        return True
    
    def bfs_path(start, end, adj, forbidden):
        """Find shortest path from start to end, interior vertices not in forbidden"""
        if start == end:
            return [start]
        if end in forbidden and end not in (start,):
            # end is a branch, that's fine
            pass
        
        from collections import deque
        queue = deque()
        queue.append((start, [start]))
        visited = {start}
        
        while queue:
            node, path = queue.popleft()
            if node == end:
                return path
            for neighbor in adj[node]:
                if neighbor in visited:
                    continue
                # Interior vertices cannot be branches
                if neighbor != start and neighbor != end and neighbor in forbidden:
                    continue
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
        
        return None
    
    # Try from largest clique size down
    # First, try the maximum clique
    if can_subdivide(max_clique):
        branches = max_clique
        k = len(branches)
        paths = []
        for i, j in combinations(range(k), 2):
            path = bfs_path(branches[i], branches[j], adj, set(branches))
            if path is None:
                # Shouldn't happen if can_subdivide returned True
                return {'branches': [0, 1], 'paths': [[0, 1]]}
            paths.append(path)
        return {'branches': list(branches), 'paths': paths}
    
    # Try smaller sets - use greedy approach
    # Start with a large set and remove vertices that cause issues
    best_result = None
    
    # Try all subsets of size max_k-1, max_k-2, etc.
    # This is expensive, so use a heuristic
    
    # Greedy: start with max_clique, try to find valid subdivision
    # If not, remove one vertex at a time
    
    # Better approach: try all k-cliques in the graph
    # But that's expensive. Let's use a smarter approach.
    
    # For each possible k from max_k down to 2:
    for k in range(max_k, 1, -1):
        if time.time() > deadline:
            break
        
        # Try to find k branch vertices that can be subdivided
        # Use the max clique as a starting point
        if k == max_k:
            candidates = [max_clique]
        else:
            # Generate subsets of max_clique of size k
            candidates = list(combinations(max_clique, k))
            # Also try other k-cliques
            # For efficiency, just use subsets of max_clique
        
        for cand in candidates:
            if time.time() > deadline:
                break
            if can_subdivide(list(cand)):
                branches = list(cand)
                k = len(branches)
                paths = []
                for i, j in combinations(range(k), 2):
                    path = bfs_path(branches[i], branches[j], adj, set(branches))
                    if path is None:
                        continue
                    paths.append(path)
                if len(paths) == k*(k-1)//2:
                    return {'branches': branches, 'paths': paths}
    
    # Fallback: find any valid subdivision
    # Try pairs
    for u, v in edges:
        return {'branches': [u, v], 'paths': [[u, v]]}
    
    return {'branches': [0, 1], 'paths': [[0, 1]]}

# EVOLVE-BLOCK-END
