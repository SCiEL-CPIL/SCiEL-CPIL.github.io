"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list and set for O(1) lookup
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Convert to sorted lists for consistent iteration
    adj_list = [sorted(adj[i]) for i in range(n)]
    
    # Calculate degrees
    degrees = [len(adj_list[i]) for i in range(n)]
    
    # Global deadline
    global_deadline = time.time() + 85.0  # 5s safety margin
    
    def find_shortest_paths(u, v, forbidden, max_paths=10):
        """
        Find shortest simple paths from u to v avoiding forbidden internal nodes.
        Uses BFS to find shortest paths first.
        """
        if u == v:
            return []
        
        # BFS to find shortest path length
        # State: (current_node, path)
        # We want all shortest paths, then slightly longer if needed
        
        best_len = None
        shortest_paths = []
        
        # BFS layer by layer
        from collections import deque
        queue = deque()
        queue.append((u, [u]))
        visited = {u}
        
        found_target = False
        current_layer_paths = []
        
        while queue and not found_target:
            layer_size = len(queue)
            next_layer_paths = []
            
            for _ in range(layer_size):
                current, path = queue.popleft()
                
                if current == v:
                    if len(path) > 1:
                        if best_len is None:
                            best_len = len(path)
                        if len(path) == best_len:
                            shortest_paths.append(path)
                    found_target = True
                    continue
                
                for neighbor in adj_list[current]:
                    if neighbor == v:
                        # Can go to target
                        new_path = path + [neighbor]
                        if best_len is None:
                            best_len = len(new_path)
                        if len(new_path) == best_len:
                            shortest_paths.append(new_path)
                        found_target = True
                    elif neighbor not in visited and neighbor not in forbidden and len(path) < 6:
                        visited.add(neighbor)
                        new_path = path + [neighbor]
                        queue.append((neighbor, new_path))
            
            if found_target:
                break
        
        if shortest_paths:
            # Sort by length (should all be same length)
            shortest_paths.sort(key=len)
            return shortest_paths[:max_paths]
        
        # If no short path found, try DFS for longer paths
        paths = []
        
        def dfs(current, target, path, visited_set):
            if len(paths) >= max_paths:
                return
            if current == target:
                if len(path) > 1:
                    paths.append(list(path))
                return
            
            for neighbor in adj_list[current]:
                if neighbor == target:
                    dfs(neighbor, target, path + [neighbor], visited_set | {neighbor})
                elif neighbor not in visited_set and neighbor not in forbidden and len(path) < 8:
                    dfs(neighbor, target, path + [neighbor], visited_set | {neighbor})
        
        dfs(u, v, [u], {u})
        paths.sort(key=len)
        return paths[:max_paths]
    
    def check_clique_subdivision(branches):
        """
        Check if the given set of branches supports a clique subdivision.
        Returns list of paths if found, else None.
        """
        k = len(branches)
        if k < 2:
            return None
        
        branch_set = set(branches)
        pairs = list(combinations(branches, 2))
        
        # Order pairs: prioritize pairs that are NOT directly connected (more constrained)
        # Actually, let's order by whether they have a direct edge first (easier to satisfy)
        # Or order by the number of common neighbors (fewer = more constrained)
        
        def pair_constraint_score(pair):
            u, v = pair
            if u in adj[v]:
                return 0  # Direct edge, easy
            # Count common neighbors
            common = len(adj[u] & adj[v])
            return -common  # Fewer common neighbors = higher priority (more constrained)
        
        pairs.sort(key=pair_constraint_score)
        
        # Backtracking search
        best_result = None
        local_deadline = min(time.time() + 2.0, global_deadline)
        
        def solve(idx, used_internal, current_paths):
            nonlocal best_result
            if time.time() > local_deadline:
                return False
            
            if idx == len(pairs):
                best_result = list(current_paths)
                return True
            
            u, v = pairs[idx]
            
            # Forbidden: used_internal + other branches
            forbidden = used_internal | (branch_set - {u, v})
            
            candidates = find_shortest_paths(u, v, forbidden, max_paths=8)
            
            if not candidates:
                return False
            
            # Try candidates, shortest first
            for path in candidates:
                internal_nodes = set(path[1:-1])
                
                # Verify path is valid
                valid = True
                for i in range(len(path) - 1):
                    if path[i+1] not in adj[path[i]]:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Check internal nodes don't conflict
                if internal_nodes & used_internal:
                    continue
                
                new_used_internal = used_internal | internal_nodes
                new_paths = current_paths + [path]
                
                if solve(idx + 1, new_used_internal, new_paths):
                    return True
            
            return False
        
        if solve(0, set(), []):
            return best_result
        
        return None
    
    # Strategy: Try to find large clique subdivisions
    # Use graph structure to identify good candidates
    
    # Compute a "centrality" score: degree + number of high-degree neighbors
    def centrality_score(v):
        score = degrees[v]
        for nb in adj_list[v]:
            score += degrees[nb] * 0.1
        return score
    
    centrality = [centrality_score(i) for i in range(n)]
    candidate_indices = sorted(range(n), key=lambda i: centrality[i], reverse=True)
    
    # Also try to find actual cliques in the graph as starting points
    # A clique of size c in the graph can potentially be extended to a clique subdivision
    
    def find_max_clique(vertices):
        """Find maximum clique in the induced subgraph on vertices."""
        if not vertices:
            return []
        
        best_clique = []
        vertices = list(vertices)
        
        def expand(clique, candidates):
            nonlocal best_clique
            if len(clique) > len(best_clique):
                best_clique = list(clique)
            
            if not candidates:
                return
            
            for i, v in enumerate(candidates):
                if all(v in adj[u] for u in clique):
                    new_clique = clique + [v]
                    new_candidates = [c for c in candidates[i+1:] if c in adj[v]]
                    expand(new_clique, new_candidates)
        
        expand([], vertices)
        return best_clique
    
    # Find max clique in the whole graph
    max_clique = find_max_clique(range(n))
    
    # Build candidate pool: union of max clique and top centrality vertices
    clique_set = set(max_clique)
    top_central = set(candidate_indices[:15])
    candidate_pool = list(clique_set | top_central)
    candidate_pool.sort(key=lambda i: centrality[i], reverse=True)
    
    best_solution = None
    best_k = 0
    
    # Try k from 8 down to 2
    for k in range(8, 1, -1):
        if time.time() > global_deadline - 5.0:
            break
        
        if k > len(candidate_pool):
            continue
        
        # Use different pools for different k
        if k <= 6:
            pool = candidate_pool[:min(14, len(candidate_pool))]
        elif k == 7:
            pool = candidate_pool[:min(12, len(candidate_pool))]
        else:
            pool = candidate_pool[:min(10, len(candidate_pool))]
        
        found = False
        for subset in combinations(pool, k):
            if time.time() > global_deadline - 3.0:
                break
            
            branches = list(subset)
            paths = check_clique_subdivision(branches)
            
            if paths is not None:
                best_solution = {
                    'branches': branches,
                    'paths': paths
                }
                best_k = k
                found = True
                break
        
        if found:
            break
    
    # If we didn't find anything with the main search, try smaller k with larger pool
    if best_solution is None:
        for k in range(5, 1, -1):
            if time.time() > global_deadline - 3.0:
                break
            
            pool = candidate_pool[:min(18, len(candidate_pool))]
            for subset in combinations(pool, k):
                if time.time() > global_deadline - 2.0:
                    break
                
                branches = list(subset)
                paths = check_clique_subdivision(branches)
                
                if paths is not None:
                    best_solution = {
                        'branches': branches,
                        'paths': paths
                    }
                    break
            
            if best_solution is not None:
                break
    
    # Fallback: find any 2 branches with a path
    if best_solution is None:
        # Find an edge
        for u, v in edges:
            if u in adj[v]:
                best_solution = {
                    'branches': [u, v],
                    'paths': [[u, v]]
                }
                break
        
        if best_solution is None:
            # Should not happen for a connected graph
            best_solution = {
                'branches': [0, 1],
                'paths': [[0, 1]]
            }
    
    # Ensure types are correct
    best_solution['branches'] = [int(b) for b in best_solution['branches']]
    best_solution['paths'] = [[int(v) for v in path] for path in best_solution['paths']]
    
    return best_solution

# EVOLVE-BLOCK-END
