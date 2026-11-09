import time
import itertools
import sys

def construct():
    # Graph data
    n = 24
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        edge_set.add((u, v))
        edge_set.add((v, u))

    # Sort vertices by degree descending to prioritize high-degree nodes as branches
    degrees = [len(adj[i]) for i in range(n)]
    vertices_by_degree = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    
    # We will try to find K_k subdivisions for k = 6, 5, 4
    # Candidate branches will be drawn from the top 'top_k' vertices
    top_k = 12
    
    def find_simple_paths(u, v, forbidden, max_paths=20):
        """
        Find up to max_paths simple paths from u to v avoiding forbidden vertices.
        forbidden includes branch vertices and previously used internal vertices.
        """
        paths = []
        # DFS
        stack = [(u, [u])]
        while stack:
            node, path = stack.pop()
            if node == v:
                paths.append(path)
                if len(paths) >= max_paths:
                    break
                continue
            
            for nei in adj[node]:
                if nei not in forbidden:
                    # Avoid cycles
                    if nei not in path:
                        stack.append((nei, path + [nei]))
        
        # Sort by length to prefer shorter paths (greedy heuristic)
        paths.sort(key=len)
        return paths

    def check_subdivision(branches):
        """
        Check if the given set of branches forms a K_k subdivision.
        Returns a dict of paths if successful, else None.
        """
        k = len(branches)
        branch_set = set(branches)
        pairs = list(itertools.combinations(branches, 2))
        
        # Initial forbidden set: all branch vertices
        # Internal vertices cannot be branches
        initial_forbidden = set(branch_set)
        
        # We will assign paths to pairs.
        # State: index of next pair to assign, set of used internal vertices, list of found paths
        # To allow backtracking, we use recursion.
        
        best_paths = [None]
        
        def backtrack(pair_idx, used_internal, current_paths):
            if pair_idx == len(pairs):
                # All pairs assigned
                best_paths[0] = current_paths[:]
                return True
            
            # Pruning: check if enough vertices remain
            # Each remaining pair needs at least 1 internal vertex (if not direct edge) 
            # Actually, if direct edge exists, 0 internal vertices needed.
            # But we don't know which pairs will use direct edges yet.
            # Conservative prune: if len(used_internal) > n - k - (len(pairs) - pair_idx):
            # This is not quite right because direct edges use 0.
            # Let's just rely on the search.
            
            u, v = pairs[pair_idx]
            
            # Forbidden for this path: branch_set + used_internal
            forbidden = initial_forbidden.union(used_internal)
            
            # Find paths from u to v
            # Note: u and v are in forbidden, but we start at u and end at v.
            # The DFS in find_simple_paths handles start/end correctly if we 
            # ensure we don't block u and v from being endpoints.
            # My find_simple_paths checks `if nei not in forbidden`.
            # If u is in forbidden, we start there. The neighbors of u are checked.
            # If v is in forbidden, we stop when node == v.
            # So it works.
            
            paths = find_simple_paths(u, v, forbidden)
            
            # Try each path
            for p in paths:
                # Internal vertices are p[1:-1]
                internal = set(p[1:-1])
                
                # If internal vertices conflict with used_internal, skip (shouldn't happen due to forbidden)
                if internal & used_internal:
                    continue
                
                # Check if any internal vertex is a branch (shouldn't happen due to forbidden)
                if internal & branch_set:
                    continue
                
                # Recurse
                new_used = used_internal.union(internal)
                if backtrack(pair_idx + 1, new_used, current_paths + [p]):
                    return True
            
            return False
        
        # Start backtracking
        if backtrack(0, set(), []):
            return best_paths[0]
        else:
            return None

    # Try to find largest k
    best_solution = None
    best_k = 0
    
    # Deadline guard
    start_time = time.time()
    deadline = 85.0 # Leave 5s margin
    
    # Try k from 6 down to 4
    # We can also try 3 if nothing else works, but seed does that.
    for k in range(6, 3, -1):
        if time.time() > deadline:
            break
            
        # Generate candidate sets from top_k vertices
        # To reduce combinations, we can fix the first few high-degree vertices
        # But for now, let's iterate over combinations of top_k
        candidates = list(itertools.combinations(vertices_by_degree[:top_k], k))
        
        # Shuffle candidates to avoid bias towards first ones? 
        # Or sort by sum of degrees?
        # Let's sort by sum of degrees descending
        candidates.sort(key=lambda c: sum(degrees[v] for v in c), reverse=True)
        
        for cand in candidates:
            if time.time() > deadline:
                break
                
            result = check_subdivision(cand)
            if result:
                best_k = k
                best_solution = {
                    'branches': list(cand),
                    'paths': [list(p) for p in result]
                }
                # If we found K_6, we can stop? 
                # The task is to maximize. K_6 > K_5 > K_4.
                # So if we find K_6, it's the best possible for this graph size likely?
                # Max K_k subdivision in n=24. 
                # If we find K_6, we return it immediately as it's the max we're trying.
                # But wait, could there be a K_7? 
                # We are trying 6, 5, 4. If we find 6, we stop and return.
                # If we want to be safe, we could try 7, but combinations explode.
                # Let's stick to 6, 5, 4.
                break
        
        if best_k > 0:
            break

    # Fallback to seed behavior (find a triangle)
    if not best_solution:
        # Find any triangle
        found = False
        for u, v, w in itertools.combinations(range(n), 3):
            if (u, v) in edge_set and (v, w) in edge_set and (u, w) in edge_set:
                best_solution = {
                    'branches': [u, v, w],
                    'paths': [[u, v], [v, w], [u, w]]
                }
                found = True
                break
        
        if not found:
            # Fallback to edge
            u, v = edges[0]
            best_solution = {
                'branches': [u, v],
                'paths': [[u, v]]
            }
            
    # Ensure types are correct
    # branches: list of ints
    # paths: list of lists of ints
    
    return best_solution

P = {'n': 24, 'edges': [[0, 1], [0, 3], [0, 5], [0, 9], [0, 10], [0, 12], [0, 20], [0, 22], [1, 4], [1, 5], [1, 6], [1, 14], [1, 15], [1, 17], [1, 19], [1, 20], [1, 22], [2, 3], [2, 5], [2, 7], [2, 8], [3, 7], [3, 9], [3, 10], [3, 11], [3, 13], [3, 14], [3, 16], [3, 19], [3, 20], [4, 6], [4, 9], [4, 11], [4, 12], [4, 13], [4, 14], [4, 15], [4, 18], [4, 19], [4, 20], [4, 21], [4, 23], [5, 7], [5, 8], [5, 9], [5, 12], [5, 14], [5, 15], [5, 17], [5, 18], [5, 19], [5, 20], [5, 21], [6, 10], [6, 13], [6, 16], [6, 20], [6, 21], [6, 22], [7, 15], [7, 16], [7, 19], [7, 22], [8, 9], [8, 15], [8, 18], [8, 20], [8, 22], [9, 11], [9, 12], [9, 15], [9, 17], [9, 21], [9, 23], [10, 12], [11, 13], [11, 14], [11, 17], [11, 18], [11, 20], [11, 22], [12, 13], [12, 15], [12, 16], [12, 22], [12, 23], [13, 16], [14, 16], [14, 17], [14, 18], [14, 19], [15, 16], [15, 19], [15, 20], [15, 22], [16, 18], [16, 21], [17, 22], [18, 19], [18, 20], [19, 21], [20, 22], [20, 23], [21, 22], [21, 23], [22, 23]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    adj = [[] for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        edge_set.add((u, v))
        edge_set.add((v, u))

    degrees = [len(adj[i]) for i in range(n)]
    vertices_by_degree = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    
    top_k = 12
    
    def find_simple_paths(u, v, forbidden, max_paths=50):
        paths = []
        stack = [(u, [u])]
        while stack:
            node, path = stack.pop()
            if node == v:
                paths.append(path)
                if len(paths) >= max_paths:
                    break
                continue
            
            for nei in adj[node]:
                if nei not in forbidden and nei not in path:
                    stack.append((nei, path + [nei]))
        
        paths.sort(key=len)
        return paths

    def check_subdivision(branches):
        k = len(branches)
        branch_set = set(branches)
        pairs = list(itertools.combinations(branches, 2))
        initial_forbidden = set(branch_set)
        best_paths = [None]
        
        def backtrack(pair_idx, used_internal, current_paths):
            if pair_idx == len(pairs):
                best_paths[0] = current_paths[:]
                return True
            
            u, v = pairs[pair_idx]
            forbidden = initial_forbidden.union(used_internal)
            
            paths = find_simple_paths(u, v, forbidden)
            
            for p in paths:
                internal = set(p[1:-1])
                if internal & used_internal:
                    continue
                if internal & branch_set:
                    continue
                
                new_used = used_internal.union(internal)
                if backtrack(pair_idx + 1, new_used, current_paths + [p]):
                    return True
            
            return False
        
        if backtrack(0, set(), []):
            return best_paths[0]
        return None

    best_solution = None
    best_k = 0
    
    start_time = time.time()
    deadline = 85.0
    
    for k in range(6, 3, -1):
        if time.time() > deadline:
            break
            
        candidates = list(itertools.combinations(vertices_by_degree[:top_k], k))
        candidates.sort(key=lambda c: sum(degrees[v] for v in c), reverse=True)
        
        for cand in candidates:
            if time.time() > deadline:
                break
                
            result = check_subdivision(cand)
            if result:
                best_k = k
                best_solution = {
                    'branches': list(cand),
                    'paths': [list(p) for p in result]
                }
                break
        
        if best_k > 0:
            break

    if not best_solution:
        for u, v, w in itertools.combinations(range(n), 3):
            if (u, v) in edge_set and (v, w) in edge_set and (u, w) in edge_set:
                best_solution = {
                    'branches': [u, v, w],
                    'paths': [[u, v], [v, w], [u, w]]
                }
                break
        
        if not best_solution:
            u, v = edges[0]
            best_solution = {
                'branches': [u, v],
                'paths': [[u, v]]
            }
            
    return best_solution
