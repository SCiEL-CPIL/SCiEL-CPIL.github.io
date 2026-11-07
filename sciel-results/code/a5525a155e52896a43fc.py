import time
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _bfs_diameter(n, adj):
    max_d = 0
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        queue = [s]
        qi = 0
        while qi < len(queue):
            u = queue[qi]
            qi += 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        if -1 in dist:
            return -1
        md = max(dist)
        if md > max_d:
            max_d = md
    return max_d

def _is_triangle_free(n, edges):
    adj = _build_adj(n, edges)
    for u in range(n):
        for v in adj[u]:
            if v <= u:
                continue
            for w in adj[v]:
                if w > v and w in adj[u]:
                    return False
    return True

def _valid_solution(n, base_edges, added_edges):
    all_edges = base_edges + added_edges
    edge_set = set()
    for u, v in all_edges:
        a, b = min(u, v), max(u, v)
        if (a, b) in edge_set:
            return False
        edge_set.add((a, b))
    if not _is_triangle_free(n, all_edges):
        return False
    adj = _build_adj(n, all_edges)
    diam = _bfs_diameter(n, adj)
    return diam != -1 and diam <= 4

def construct():
    n = P['n']
    base_edges = P['edges']
    base_edge_set = set((min(u, v), max(u, v)) for u, v in base_edges)
    
    deadline = time.time() + 85
    
    # Only add edges between even and odd vertices (bipartite => triangle-free)
    even_verts = list(range(0, n, 2))
    odd_verts = list(range(1, n, 2))
    
    # Generate all candidate edges, sorted by "usefulness" (longer = better)
    candidates = []
    for u in even_verts:
        for v in odd_verts:
            key = (min(u, v), max(u, v))
            if key not in base_edge_set:
                candidates.append((u, v))
    
    # Sort by span descending - longer edges are more useful for reducing diameter
    candidates.sort(key=lambda e: abs(e[0] - e[1]), reverse=True)
    
    # Try solutions with increasing number of edges
    # Start with small k and go up
    
    # First, try a structured approach: place edges at regular intervals
    # For diameter 4, we need good coverage
    
    # Let's try k=5, 6, 7, 8 with smart candidate selection
    
    best_sol = None
    
    # Try structured solutions first
    # Idea: divide path into segments, add long edges between segment endpoints
    structured_attempts = []
    
    # Attempt: edges that create a "star" pattern from a central hub
    # Hub at 11 or 12 (center)
    for hub in [11, 12]:
        hub_parity = hub % 2
        edges = []
        for other in range(n):
            if other == hub:
                continue
            if other % 2 != hub_parity:  # opposite parity
                key = (min(hub, other), max(hub, other))
                if key not in base_edge_set:
                    edges.append((hub, other))
        structured_attempts.append(edges)
    
    # Attempt: two hubs
    for h1 in [5, 6, 7]:
        for h2 in [16, 17, 18]:
            if h1 % 2 == h2 % 2:
                continue
            edges = [(h1, h2)]
            # Connect vertices to nearest hub
            for v in range(n):
                if v == h1 or v == h2:
                    continue
                if v % 2 != h1 % 2:
                    key = (min(h1, v), max(h1, v))
                    if key not in base_edge_set and key not in set((min(a,b),max(a,b)) for a,b in edges):
                        edges.append((h1, v))
                if v % 2 != h2 % 2:
                    key = (min(h2, v), max(h2, v))
                    if key not in base_edge_set and key not in set((min(a,b),max(a,b)) for a,b in edges):
                        edges.append((h2, v))
            structured_attempts.append(edges)
    
    # Try each structured attempt
    for attempt in structured_attempts:
        if time.time() > deadline - 10:
            break
        # Reduce the attempt
        reduced = list(attempt)
        for i in range(len(reduced) - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            test = reduced[:i] + reduced[i+1:]
            if _valid_solution(n, base_edges, test):
                reduced = test
        
        if _valid_solution(n, base_edges, reduced):
            if best_sol is None or len(reduced) < len(best_sol):
                best_sol = reduced
    
    # If we found a solution, try to improve with local search
    if best_sol is not None:
        # Try to replace edges with better ones
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            current = list(best_sol)
            current_set = set((min(u,v), max(u,v)) for u,v in current)
            
            for i in range(len(current)):
                # Try replacing current[i] with another edge
                for new_edge in candidates[:100]:
                    key_new = (min(new_edge[0], new_edge[1]), max(new_edge[0], new_edge[1]))
                    if key_new in current_set:
                        continue
                    test = current[:i] + [new_edge] + current[i+1:]
                    test_set = set((min(u,v), max(u,v)) for u,v in test)
                    if len(test_set) != len(test):
                        continue
                    if _valid_solution(n, base_edges, test):
                        # Check if this is an improvement (it's same size, but maybe enables removal)
                        best_sol = test
                        improved = True
                        break
                if improved:
                    break
    
    # If no solution found yet, use greedy
    if best_sol is None:
        added = []
        added_set = set()
        current_edges = list(base_edges)
        
        for iteration in range(15):
            if time.time() > deadline - 5:
                break
            
            adj = _build_adj(n, current_edges)
            diam = _bfs_diameter(n, adj)
            
            if diam <= 4:
                break
            if diam == -1:
                break
            
            # Find the edge that most reduces diameter
            best_edge = None
            best_diam = diam
            
            for edge in candidates:
                u, v = edge
                key = (min(u, v), max(u, v))
                if key in added_set:
                    continue
                
                # Quick triangle check
                common = adj[u] & adj[v]
                if common:
                    continue
                
                adj[u].add(v)
                adj[v].add(u)
                new_diam = _bfs_diameter(n, adj)
                adj[u].remove(v)
                adj[v].remove(u)
                
                if new_diam != -1 and new_diam < best_diam:
                    best_diam = new_diam
                    best_edge = edge
            
            if best_edge is None:
                break
            
            added.append(best_edge)
            added_set.add((min(best_edge[0], best_edge[1]), max(best_edge[0], best_edge[1])))
            current_edges.append(best_edge)
        
        # Reduce
        for i in range(len(added) - 1, -1, -1):
            if time.time() > deadline - 2:
                break
            reduced = added[:i] + added[i+1:]
            if _valid_solution(n, base_edges, reduced):
                added = reduced
        
        if _valid_solution(n, base_edges, added):
            best_sol = added
    
    if best_sol is not None:
        return [[u, v] for u, v in best_sol]
    
    return []
