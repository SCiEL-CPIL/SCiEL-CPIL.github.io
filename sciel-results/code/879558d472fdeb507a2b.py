"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

P = {'n': 16}

def _check_diameter_two(n, edge_set, removed_edge=None):
    """Check if graph has diameter exactly 2. Returns True if valid."""
    adj = [set() for _ in range(n)]
    for (u, v) in edge_set:
        if (u, v) == removed_edge or (v, u) == removed_edge:
            continue
        adj[u].add(v)
        adj[v].add(u)
    
    # Check connectivity and diameter <= 2
    for start in range(n):
        if start == 0:
            continue  # BFS from 0
        # BFS from start
        dist = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        for i in range(n):
            if dist[i] == -1:
                return False  # disconnected
            if i != start and dist[i] > 2:
                return False  # diameter > 2
    return True

def _is_critical(n, edge_set, edge):
    """Check if removing this edge from edge_set makes diameter > 2 or disconnects."""
    # First check: without this edge, is there still a path of length <= 2?
    u, v = edge
    # Check if there's a common neighbor
    adj = [set() for _ in range(n)]
    for (a, b) in edge_set:
        if (a, b) == (u, v) or (a, b) == (v, u):
            continue
        adj[a].add(b)
        adj[b].add(a)
    
    # Direct edge check: u and v are not adjacent (we removed it)
    # Check for path of length 2
    if adj[u] & adj[v]:
        return False  # There's a common neighbor, so path of length 2 exists
    
    # Check if graph is still connected with diameter <= 2
    # Quick check: BFS from u to v
    dist = [-1] * n
    dist[u] = 0
    queue = deque([u])
    while queue:
        x = queue.popleft()
        if dist[x] >= 2:
            continue
        for y in adj[x]:
            if dist[y] == -1:
                dist[y] = dist[x] + 1
                queue.append(y)
    
    if dist[v] == -1:
        return True  # disconnected, so critical
    if dist[v] > 2:
        return True  # path longer than 2, so critical
    
    # Full diameter check needed
    return not _check_diameter_two(n, edge_set, removed_edge=(u, v))

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    n = P['n']
    
    # Generate all possible edges
    all_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.append((i, j))
    
    # Shuffle for randomness
    random.seed(42)
    random.shuffle(all_edges)
    
    # Greedy: try to add each edge if it's critical
    edge_set = set()
    
    for edge in all_edges:
        if time.time() > deadline:
            break
        # Try adding this edge
        new_set = edge_set | {edge}
        if _is_critical(n, new_set, edge):
            edge_set.add(edge)
    
    # Convert to list of lists
    edges = [[u, v] for (u, v) in edge_set]
    
    # Verify the result
    if not _check_diameter_two(n, edge_set):
        # Fallback to K_{8,8}
        half = n // 2
        edges = []
        for u in range(half):
            for v in range(half, n):
                edges.append([u, v])
    
    return edges
# EVOLVE-BLOCK-END
