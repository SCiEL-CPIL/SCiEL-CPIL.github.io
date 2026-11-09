import time
import itertools
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Sort vertices by degree
    vertices_by_degree = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
    
    def find_paths_for_branches(B):
        """
        Finds internally disjoint paths for all pairs in B.
        Returns a dict mapping (u, v) to path list, or None if not found.
        """
        k = len(B)
        if k < 2:
            return None
            
        pairs = list(itertools.combinations(B, 2))
        # Sort pairs to try "harder" ones first? Or just standard order.
        # Heuristic: Pairs with fewer direct connections might be harder.
        # Let's just use standard order for now, or sort by distance.
        
        # State for backtracking
        used_vertices = set()
        paths = {}
        
        # Remove branch vertices from consideration for internal use
        B_set = set(B)
        
        def dfs(idx):
            if idx == len(pairs):
                return True
                
            if time.time() > deadline:
                return False
                
            u, v = pairs[idx]
            
            # Find a path from u to v using only vertices not in used_vertices and not in B
            # We need to find a simple path.
            # Since N is small, we can do a DFS to find *any* path, but we want to find one that allows future paths.
            # A simple strategy: Find the shortest path? Or try all paths?
            # Trying all paths is exponential.
            # Let's find the shortest path first. If it fails, try longer?
            # Actually, finding *a* path is easy. The constraint is disjointness.
            # We can use a BFS to find the shortest path in the available graph.
            
            available = [x for x in range(n) if x not in used_vertices and x not in B_set]
            
            # BFS from u to v
            parent = {u: None}
            queue = deque([u])
            found = False
            while queue and not found:
                curr = queue.popleft()
                if curr == v:
                    found = True
                    break
                for neighbor in adj[curr]:
                    if neighbor in parent:
                        continue
                    if neighbor in B_set and neighbor not in (u, v):
                        continue
                    if neighbor in used_vertices:
                        continue
                    parent[neighbor] = curr
                    queue.append(neighbor)
            
            if not found:
                return False
            
            # Reconstruct path
            path = []
            curr = v
            while curr is not None:
                path.append(curr)
                curr = parent[curr]
            path.reverse()
            
            # Internal vertices
            internal = path[1:-1]
            
            # Check if any internal vertex is already used
            # BFS ensures we only used available vertices, so this should be safe.
            # But wait, BFS finds *one* path. It might block future paths.
            # We should try to find a path that minimizes impact?
            # For now, let's assume shortest path is a good heuristic.
            
            # Record path
            paths[(u, v)] = path
            used_vertices.update(internal)
            
            if dfs(idx + 1):
                return True
            
            # Backtrack
            used_vertices.difference_update(internal)
            del paths[(u, v)]
            
            # Optimization: If shortest path failed to lead to a solution, 
            # trying a longer path might work, but is expensive.
            # Given the small size, let's stick to shortest path for now.
            # If this returns False, we might miss a valid subdivision.
            # To improve, we could iterate through all simple paths up to a certain length.
            # But let's try this first.
            
            return False
        
        if dfs(0):
            return paths
        return None

    # Try to find largest k
    best_result = None
    
    # Max k is limited by N and time
    # Try k from 6 down to 2
    for k in range(6, 1, -1):
        if time.time() > deadline:
            break
            
        # Candidate sets from top 12 vertices
        # If k is large, we need to be more selective
        candidates = vertices_by_degree[:12]
        
        for B in itertools.combinations(candidates, k):
            if time.time() > deadline:
                break
                
            # Quick check: degree of each branch must be at least k-1
            # because each branch needs to connect to k-1 other branches via distinct first steps
            valid = True
            for v in B:
                if len(adj[v]) < k - 1:
                    valid = False
                    break
            if not valid:
                continue
                
            paths = find_paths_for_branches(B)
            if paths:
                # Convert to required format
                branch_list = list(B)
                path_list = []
                for (u, v) in itertools.combinations(branch_list, 2):
                    path_list.append(list(paths[(u, v)]))
                
                best_result = {
                    'branches': branch_list,
                    'paths': path_list
                }
                break
        
        if best_result:
            break

    if best_result is None:
        # Fallback: find an edge
        if edges:
            u, v = edges[0]
            best_result = {
                'branches': [u, v],
                'paths': [[u, v]]
            }
        else:
            # Should not happen
            best_result = {
                'branches': [0, 1],
                'paths': [[0, 1]]
            }
            
    return best_result

# EVOLVE-BLOCK-END
