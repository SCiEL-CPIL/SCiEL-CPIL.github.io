"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    t0 = time.time()
    deadline = t0 + 80.0  # 10s safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Build adjacency matrix for fast lookup
    adjmat = [[False] * n for _ in range(n)]
    for u, v in edges:
        adjmat[u][v] = True
        adjmat[v][u] = True
    
    def bfs_path(src, dst, forbidden):
        """Find shortest path from src to dst avoiding forbidden vertices (except endpoints)."""
        if src == dst:
            return [src]
        if dst in forbidden:
            return None
        visited = {src}
        parent = {src: None}
        queue = deque([src])
        while queue:
            cur = queue.popleft()
            if cur == dst:
                # Reconstruct path
                path = []
                node = dst
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return path
            for nb in adj[cur]:
                if nb not in visited and nb not in forbidden:
                    visited.add(nb)
                    parent[nb] = cur
                    queue.append(nb)
        return None
    
    def try_route(branches, adjmat_copy):
        """Try to route all pairs of branches with internally disjoint paths.
        Returns (success, paths_dict) where paths_dict maps (u,v) to path."""
        k = len(branches)
        pairs = list(combinations(range(k), 2))
        
        # Track used internal vertices
        used_internal = set()
        paths = {}
        
        # Try routing in order of "difficulty" - pairs that are farther apart first
        # Actually, let's try all pairs and use backtracking if needed
        # For efficiency, try a greedy order: pairs with fewer available paths first
        
        # Compute for each pair, the number of disjoint paths available
        # This is complex; let's just try in a heuristic order
        
        # Order pairs by "distance" - shorter paths are easier to find but might block others
        # Let's try: pairs that share no vertices in their shortest path with other pairs first
        
        # Simple approach: try each pair, find a path, mark internal vertices as used
        # If stuck, backtrack
        
        def backtrack(idx):
            if time.time() > deadline:
                return False, None
            
            if idx == len(pairs):
                return True, paths
            
            i, j = pairs[idx]
            b_i = branches[i]
            b_j = branches[j]
            
            # Find a path from b_i to b_j avoiding used_internal and other branches
            forbidden = used_internal | set(branches)
            forbidden.discard(b_i)
            forbidden.discard(b_j)
            
            # Find all possible shortest paths and try each
            # For efficiency, just find one path first, then if it fails later, try alternatives
            
            path = bfs_path(b_i, b_j, forbidden)
            if path is None:
                return False, None
            
            # Mark internal vertices
            internal = set(path[1:-1])
            used_internal.update(internal)
            paths[(i, j)] = path
            
            ok, _ = backtrack(idx + 1)
            if ok:
                return True, paths
            
            # Backtrack
            used_internal.difference_update(internal)
            del paths[(i, j)]
            
            # Try finding an alternative path (not just shortest)
            # Do a DFS to find a different path
            def find_alt_path(src, dst, forb, exclude_edges=None):
                """Find a path from src to dst avoiding forb, optionally excluding certain edges."""
                if exclude_edges is None:
                    exclude_edges = set()
                visited = {src}
                parent = {src: None}
                parent_edge = {src: None}
                queue = deque([src])
                while queue:
                    cur = queue.popleft()
                    if cur == dst:
                        # Reconstruct
                        path = []
                        node = dst
                        while node is not None:
                            path.append(node)
                            node = parent[node]
                        path.reverse()
                        return path
                    for nb in adj[cur]:
                        if nb not in visited and nb not in forb:
                            edge = (min(cur, nb), max(cur, nb))
                            if edge not in exclude_edges:
                                visited.add(nb)
                                parent[nb] = cur
                                parent_edge[nb] = edge
                                queue.append(nb)
                return None
            
            # Try to find an alternative path by excluding the first edge of the previous path
            if len(path) > 2:
                first_edge = (min(path[0], path[1]), max(path[0], path[1]))
                alt_path = find_alt_path(b_i, b_j, forbidden, exclude_edges={first_edge})
                if alt_path is not None and set(alt_path[1:-1]) != internal:
                    internal2 = set(alt_path[1:-1])
                    used_internal.update(internal2)
                    paths[(i, j)] = alt_path
                    ok, _ = backtrack(idx + 1)
                    if ok:
                        return True, paths
                    used_internal.difference_update(internal2)
                    del paths[(i, j)]
            
            return False, None
        
        ok, result_paths = backtrack(0)
        if ok:
            return True, result_paths
        return False, None
    
    def convert_to_output(branches, paths_dict):
        """Convert internal representation to output format."""
        k = len(branches)
        output_paths = []
        for i, j in combinations(range(k), 2):
            path = paths_dict[(i, j)]
            output_paths.append(list(path))
        return {
            'branches': [int(b) for b in branches],
            'paths': output_paths
        }
    
    # Strategy: Start with a large candidate set and try to route.
    # If routing fails, remove the "worst" vertex and retry.
    
    # First, find a good initial set of branches.
    # Use degree as a heuristic - high-degree vertices are more likely to be in a clique subdivision.
    degrees = [len(adj[i]) for i in range(n)]
    candidates = sorted(range(n), key=lambda x: -degrees[x])
    
    best_result = None
    best_size = 0
    
    # Try different sizes, starting large
    max_k = min(n, 10)  # Cap at 10 branches to keep search tractable
    
    for k in range(max_k, 1, -1):
        if time.time() > deadline:
            break
        
        # Try different subsets of top candidates
        # Use a greedy approach: start with top k, try to route, if fail remove one and retry
        
        # Try multiple starting sets
        for start_offset in range(0, min(5, n - k + 1)):
            if time.time() > deadline:
                break
            
            # Take candidates[start_offset:start_offset+k]
            branch_set = candidates[start_offset:start_offset+k]
            if len(branch_set) < k:
                continue
            
            # Try to route
            ok, paths_dict = try_route(branch_set, adjmat)
            if ok:
                result = convert_to_output(branch_set, paths_dict)
                if best_result is None or k > best_size:
                    best_result = result
                    best_size = k
                break  # Found a valid set of this size, move to next size
        
        # If we found a valid set of size k, we can stop (since we go from large to small)
        if best_size == k:
            break
    
    # If we didn't find anything, fall back to a simple edge
    if best_result is None:
        u, v = edges[0]
        best_result = {'branches': [int(u), int(v)], 'paths': [[int(u), int(v)]]}
    
    return best_result

# EVOLVE-BLOCK-END
