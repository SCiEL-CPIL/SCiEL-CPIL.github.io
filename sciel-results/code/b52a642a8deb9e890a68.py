"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _check_validity(added_edges):
    """Check if added edges produce a valid triangle-free, connected, diameter<=4 graph."""
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add((min(e[0], e[1]), max(e[0], e[1])))
    
    all_edges = set(base_edges)
    for e in added_edges:
        u, v = int(min(e[0], e[1])), int(max(e[0], e[1]))
        if (u, v) in all_edges:
            return False
        all_edges.add((u, v))
    
    adj = [[] for _ in range(n)]
    for u, v in all_edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Triangle-free check
    adj_set = [set(a) for a in adj]
    for u in range(n):
        neighbors = adj[u]
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                if neighbors[j] in adj_set[neighbors[i]]:
                    return False
    
    # Connected check via BFS
    visited = [False] * n
    visited[0] = True
    queue = deque([0])
    while queue:
        node = queue.popleft()
        for nb in adj[node]:
            if not visited[nb]:
                visited[nb] = True
                queue.append(nb)
    if not all(visited):
        return False
    
    # Diameter <= 4 check via BFS from each node
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        queue = deque([src])
        while queue:
            node = queue.popleft()
            if dist[node] >= 4:
                continue
            for nb in adj[node]:
                if dist[nb] == -1:
                    dist[nb] = dist[node] + 1
                    queue.append(nb)
        if any(d == -1 for d in dist):
            return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill

    n = 24
    base_edges = set()
    for e in P['edges']:
        base_edges.add((min(e[0], e[1]), max(e[0], e[1])))

    # Precompute valid edge candidates: even-odd pairs not in base
    # Even vertices: 0,2,4,...,22 ; Odd vertices: 1,3,5,...,23
    even_verts = list(range(0, n, 2))
    odd_verts = list(range(1, n, 2))
    
    valid_candidates = []
    for u in even_verts:
        for v in odd_verts:
            if (u, v) not in base_edges:
                valid_candidates.append((u, v))
    
    # Hub-based construction: pick an even hub, connect to all odd vertices except path neighbors
    # Try all possible hubs and find the best (fewest edges)
    best_solution = None
    best_count = n * n  # Large number
    
    # Strategy: For each even vertex as potential hub, connect to all odd vertices
    # except those already adjacent in the path. Then check if diameter <= 4.
    # If not, try adding a few more edges from other even vertices.
    
    for hub in even_verts:
        if time.time() > deadline:
            break
        
        # Connect hub to all odd vertices except path neighbors
        path_neighbors_of_hub = set()
        if hub > 0:
            path_neighbors_of_hub.add(hub - 1)
        if hub < n - 1:
            path_neighbors_of_hub.add(hub + 1)
        
        hub_edges = []
        for v in odd_verts:
            if v not in path_neighbors_of_hub:
                hub_edges.append((hub, v))
        
        # Check if just hub edges suffice
        if _check_validity(hub_edges):
            if len(hub_edges) < best_count:
                best_count = len(hub_edges)
                best_solution = [list(e) for e in hub_edges]
        
        # Try local search: swap one hub edge with another to see if we can reduce
        # Actually, hub edges are already minimal for this hub. Let's try different hubs.
    
    # If no single-hub solution found, try two-hub approach
    if best_solution is None:
        # Fallback: use hub 12
        hub = 12
        path_neighbors_of_hub = {11, 13}
        hub_edges = []
        for v in odd_verts:
            if v not in path_neighbors_of_hub:
                hub_edges.append((hub, v))
        best_solution = [list(e) for e in hub_edges]
        best_count = len(hub_edges)
    else:
        # Local search: try to improve by swapping edges
        current = [list(e) for e in best_solution]
        current_set = set((min(e[0], e[1]), max(e[0], e[1])) for e in current)
        
        for i in range(len(current)):
            if time.time() > deadline:
                break
            removed = tuple(min(current[i][0], current[i][1]), max(current[i][0], current[i][1]))
            remaining = [e for j, e in enumerate(current) if j != i]
            remaining_set = set((min(e[0], e[1]), max(e[0], e[1])) for e in remaining)
            
            for cand in valid_candidates:
                if cand in remaining_set or cand == removed:
                    continue
                test = remaining + [[int(cand[0]), int(cand[1])]]
                if _check_validity(test):
                    current = test
                    current_set = set((min(e[0], e[1]), max(e[0], e[1])) for e in current)
                    break
    
    # Ensure proper types: integers must be Python int, lists must be list
    result = []
    for e in current if best_solution is None else best_solution:
        u, v = int(e[0]), int(e[1])
        result.append([u, v])
    
    return result
# EVOLVE-BLOCK-END
