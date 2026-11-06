"""Long path finder using 2-core pruning and heuristic backtracking."""

import time

# Pinned graph data
P = {'n': 100, 'edges': [[0, 10], [0, 27], [0, 80], [1, 59], [2, 50], [2, 74], [3, 12], [3, 18], [3, 62], [3, 87], [4, 48], [4, 72], [4, 89], [5, 11], [6, 36], [8, 13], [8, 45], [8, 52], [8, 94], [9, 26], [9, 47], [9, 78], [10, 81], [11, 18], [11, 21], [11, 84], [12, 22], [12, 65], [12, 83], [13, 30], [13, 76], [15, 82], [17, 35], [19, 38], [19, 49], [19, 61], [19, 96], [20, 52], [21, 23], [21, 58], [22, 33], [22, 48], [24, 29], [24, 31], [24, 55], [24, 58], [24, 89], [25, 82], [26, 44], [26, 69], [30, 45], [30, 70], [32, 96], [33, 55], [34, 66], [35, 38], [35, 93], [36, 81], [37, 42], [37, 70], [39, 59], [40, 69], [40, 97], [41, 80], [42, 44], [42, 48], [42, 68], [43, 53], [44, 50], [46, 50], [46, 73], [46, 97], [47, 78], [48, 77], [49, 59], [50, 59], [50, 62], [51, 57], [53, 68], [53, 80], [56, 58], [58, 95], [60, 65], [60, 90], [61, 88], [62, 90], [68, 81], [69, 90], [70, 92], [74, 85], [75, 82]], 'edge_probability': [9, 500]}

def _build_adjacency():
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _compute_2core(adj, n):
    """Compute the 2-core of the graph."""
    in_core = [True] * n
    degrees = [len(adj[i]) for i in range(n)]
    queue = [i for i in range(n) if degrees[i] < 2]
    
    while queue:
        u = queue.pop()
        if not in_core[u]:
            continue
        in_core[u] = False
        for v in adj[u]:
            if in_core[v]:
                degrees[v] -= 1
                if degrees[v] < 2:
                    queue.append(v)
                    
    return in_core

def construct():
    adj = _build_adjacency()
    n = P['n']
    
    # Step 1: Prune to 2-core
    in_core = _compute_2core(adj, n)
    core_nodes = [i for i in range(n) if in_core[i]]
    
    if not core_nodes:
        # Fallback if no 2-core (should not happen for this graph)
        return [0]
        
    # Create a filtered adjacency list for the core
    core_adj = [set() for _ in range(n)]
    for u in core_nodes:
        for v in adj[u]:
            if in_core[v]:
                core_adj[u].add(v)
                
    # Step 2: Heuristic DFS on the 2-core
    # We want to find the longest simple path.
    # Heuristic: Always extend to the neighbor with the fewest unvisited neighbors (Warnsdorff's rule).
    
    best_path = []
    best_len = 0
    
    start_time = time.time()
    time_limit = 55.0
    
    # Sort core nodes by degree to find good starting points (low degree first)
    core_degrees = {u: len(core_adj[u]) for u in core_nodes}
    start_order = sorted(core_nodes, key=lambda x: (core_degrees[x], x))
    
    def get_heuristic_candidates(end_node, visited):
        cands = []
        for v in core_adj[end_node]:
            if v not in visited:
                # Count unvisited neighbors in the core
                unvisited_count = sum(1 for w in core_adj[v] if w not in visited)
                cands.append((unvisited_count, v))
        cands.sort()
        return cands

    def dfs(path, visited):
        nonlocal best_path, best_len, start_time, time_limit
        
        if time.time() - start_time > time_limit:
            return
        
        curr_len = len(path)
        if curr_len > best_len:
            best_len = curr_len
            best_path = path[:]
            
        if curr_len == len(core_nodes):
            return

        # Try extending from both ends
        # To keep it simple and effective, we'll try extending from the end with fewer options first?
        # Or just try both.
        
        end_cands = get_heuristic_candidates(path[-1], visited)
        start_cands = get_heuristic_candidates(path[0], visited)
        
        # Process end extension
        for _, v in end_cands:
            path.append(v)
            visited.add(v)
            dfs(path, visited)
            visited.remove(v)
            path.pop()
            if time.time() - start_time > time_limit:
                return
                
        # Process start extension
        for _, v in start_cands:
            path.insert(0, v)
            visited.add(v)
            dfs(path, visited)
            visited.remove(v)
            path.pop(0)
            if time.time() - start_time > time_limit:
                return

    # Try starting from each node in the core
    for s in start_order:
        if time.time() - start_time > time_limit:
            break
        
        path = [s]
        visited = {s}
        dfs(path, visited)
        
        if best_len == len(core_nodes):
            break
            
    return best_path
