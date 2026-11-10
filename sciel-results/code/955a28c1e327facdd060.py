# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def construct(k, g):
    if k != 3:
        return _lww_fallback(k, g)
    # First try to find a small cage via randomized construction
    deadline = time.time() + 110
    result = _randomized_cage(3, g, deadline)
    if result is not None:
        return result
    # Fallback to algebraic construction
    return _lww_fallback(k, g)

def _randomized_cage(k, g, deadline):
    """
    Randomized backtracking to build a k-regular graph with girth >= g.
    Uses a greedy approach: maintain a partial graph, add edges/vertices
    while ensuring no short cycles are formed.
    """
    random.seed(42)
    
    # We'll try multiple random seeds and restarts
    for attempt in range(50):
        if time.time() >= deadline:
            break
        result = _single_random_attempt(k, g, deadline, attempt)
        if result is not None:
            return result
    
    return None

def _single_random_attempt(k, g, deadline, seed_offset):
    """
    One attempt at building the graph via randomized greedy construction.
    """
    random.seed(12345 + seed_offset)
    
    adj = []  # adjacency lists
    # We'll track distances more efficiently
    
    def get_dist(u, v):
        """BFS to find distance between u and v in current graph."""
        if u == v:
            return 0
        dist = [-1] * len(adj)
        dist[u] = 0
        queue = deque([u])
        while queue:
            node = queue.popleft()
            for nb in adj[node]:
                if dist[nb] == -1:
                    dist[nb] = dist[node] + 1
                    if nb == v:
                        return dist[v]
                    queue.append(nb)
        return -1  # disconnected
    
    def can_add_edge(u, v):
        """Check if adding edge (u,v) would create a cycle of length < g."""
        if u == v:
            return False
        if v in adj[u]:
            return False
        d = get_dist(u, v)
        if d == -1:
            return True  # disconnected, safe
        return d + 1 >= g  # new cycle length = d + 1
    
    # Start with vertex 0 and its k neighbors
    adj.append([])
    for i in range(1, k + 1):
        adj.append([])
        adj[0].append(i)
        adj[i].append(0)
    
    n = k  # current number of vertices (0-indexed, so n+1 total)
    
    # Vertices that still need edges
    incomplete = deque(range(1, k + 1))
    
    while incomplete and time.time() < deadline:
        v = incomplete.popleft()
        if len(adj[v]) >= k:
            continue
        
        needed = k - len(adj[v])
        
        # Try to connect to existing vertices first (randomized order)
        candidates = [u for u in range(len(adj)) if u != v and len(adj[v]) < k]
        random.shuffle(candidates)
        
        connected = False
        for u in candidates:
            if len(adj[v]) >= k:
                break
            if can_add_edge(v, u):
                adj[v].append(u)
                adj[u].append(v)
                if len(adj[u]) < k:
                    incomplete.append(u)
                connected = True
                if needed > 1 and time.time() < deadline:
                    incomplete.append(v)
                break
        
        if len(adj[v]) < k:
            # Add new vertices
            for _ in range(k - len(adj[v])):
                new_v = len(adj)
                adj.append([])
                adj[v].append(new_v)
                adj[new_v].append(v)
                incomplete.append(new_v)
    
    # Check if we succeeded
    n = len(adj)
    for i in range(n):
        if len(adj[i]) != k:
            return None
    
    # Verify girth: no cycle shorter than g
    # Use BFS from each vertex to check minimum cycle length
    # A cycle through vertex v of length L means there's a path of length L-1 from some neighbor back to v
    # More precisely: for each edge (u,v), check if there's a path of length >= g-1 from u to v not using that edge
    # Simpler: for each vertex, do BFS and check if any neighbor is at distance < g-1 (would form cycle of length < g)
    
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for nb in adj[node]:
                if dist[nb] == -1:
                    dist[nb] = dist[node] + 1
                    if dist[nb] < g - 1:
                        # This is fine, we just need to check cycles
                        pass
                    queue.append(nb)
    
    # Better girth check: for each vertex, BFS and for each neighbor, check distance
    # Actually, the girth is the minimum cycle length. A cycle of length L exists iff
    # there's a path of length L-1 between two adjacent vertices.
    # So for each edge (u,v), we need dist(u,v) >= g-1 in the graph without that edge.
    # But that's expensive. Instead: BFS from each vertex, and for each neighbor at distance d,
    # if d+1 < g, we have a cycle of length d+1.
    
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for nb in adj[node]:
                if dist[nb] == -1:
                    dist[nb] = dist[node] + 1
                    # If nb is a neighbor of start and dist[nb] + 1 < g, we have a short cycle
                    # But we need to be careful: the path from start to nb plus edge (start,nb) forms a cycle
                    # The cycle length is dist[nb] + 1
                    if nb in adj[start] and dist[nb] + 1 < g:
                        return None  # Short cycle found
                    queue.append(nb)
    
    # Convert to edge list
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    
    return n, edges

def _lww_fallback(k, g):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add_fn, sub_fn, mul_fn = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add_fn, sub_fn, mul_fn = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add_fn(x[c], mul_fn(y[a], x[b]))
                else:
                    y[c] = sub_fn(x[c], mul_fn(x[a], y[b]))
            yield 1 - side, tuple(y)
    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
# EVOLVE-BLOCK-END
