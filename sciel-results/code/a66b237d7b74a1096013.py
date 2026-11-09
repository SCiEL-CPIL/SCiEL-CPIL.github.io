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

    even_verts = list(range(0, n, 2))
    odd_verts = list(range(1, n, 2))
    
    # Start with the best single-hub solution (Hub 12)
    # This is known to be valid with 10 edges
    hub = 12
    path_neighbors_of_hub = {11, 13}
    current_edges = []
    for v in odd_verts:
        if v not in path_neighbors_of_hub:
            current_edges.append((hub, v))
            
    current_edges = [list(e) for e in current_edges]
    
    # Greedy removal: try to remove edges one by one if validity is maintained
    # This can reduce the count below 10
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(current_edges)):
            if time.time() > deadline:
                break
            # Try removing edge i
            test_edges = [e for j, e in enumerate(current_edges) if j != i]
            if _check_validity(test_edges):
                current_edges = test_edges
                improved = True
                break # Restart loop as indices changed
    
    # Ensure proper types: integers must be Python int, lists must be list
    result = []
    for e in current_edges:
        u, v = int(e[0]), int(e[1])
        result.append([u, v])
    
    return result
# EVOLVE-BLOCK-END
