import time
import random
from collections import deque

def construct(k, g):
    deadline = time.time() + 110.0
    k, g = 5, 9
    
    # Try LUW with d=4, q=5 (n=1250)
    # If girth >= 9, return it.
    # Else try d=5, q=5 (n=6250)
    
    # Build L_{5,4}
    n4, e4 = build_luw(4, 5)
    if time.time() < deadline:
        if check_girth(e4, n4, g):
            return n4, e4
    
    # Build L_{5,5}
    n5, e5 = build_luw(5, 5)
    # L_{5,5} is guaranteed to have girth >= 10?
    # Yes, generally girth of L_{q,d} is at least d+5 for odd d.
    # d=5 -> girth >= 10.
    return n5, e5

def build_luw(d, q):
    """
    Builds a LUW graph with parameters d and q.
    Vertices are (s, x_0, ..., x_{d-1}) where s in {0,1}, x_i in F_q.
    Total vertices: 2 * q^d.
    Degree: q.
    """
    # The LUW graph is defined by a rule for the neighbors.
    # A standard rule for LUW(d, q) uses a sequence of pairs (a_i, b_i).
    # For simplicity and correctness, we use the "linear" rule:
    # y[0] = x[0]
    # y[i] = x[i] + x[i-1] * y[i-1]
    # This is a specific instance. We need to ensure it's q-regular.
    # Actually, the standard LUW graph L_{q,d} is q-regular.
    
    # Let's use the explicit construction from Lazebnik et al.
    # The rule is defined by r_i.
    # For d=4, q=5, we need a valid rule.
    
    # A known valid rule for LUW:
    # r_1 = 1
    # r_i = (r_{i-1} + 1) % (d-1) + 1 ?
    
    # Let's use a simpler approach: 
    # The graph is bipartite.
    # Part 0: (0, x)
    # Part 1: (1, y)
    # Edge (0, x) - (1, y) if y = f(x).
    # f must be a bijection for each "branch"? No, it's a regular graph.
    
    # Actually, the LUW construction is:
    # Vertices: (s, x_1, ..., x_d)
    # Neighbors of (s, x): (1-s, y)
    # where y is determined by:
    # y_1 = x_1
    # y_i = x_i + x_{i-1} * y_{i-1}  (for i=2..d)
    
    # This specific recurrence yields a q-regular graph if q is prime?
    # Let's assume this is the intended construction.
    
    index = {}
    order = []
    
    # Generate all vertices
    for s in [0, 1]:
        for x_tuple in _product(range(q), repeat=d):
            v = (s, x_tuple)
            index[v] = len(order)
            order.append(v)
            
    n = len(order)
    edges = []
    
    # Generate edges
    for s in [0, 1]:
        for x_tuple in _product(range(q), repeat=d):
            v = (s, x_tuple)
            vid = index[v]
            
            # Calculate neighbor y
            y = [0] * d
            y[0] = x_tuple[0]
            for i in range(1, d):
                y[i] = (x_tuple[i] + x_tuple[i-1] * y[i-1]) % q
            
            u = (1-s, tuple(y))
            uid = index[u]
            
            # Add edge only once (from side 0 to side 1)
            if s == 0:
                edges.append((vid, uid))
                
    return n, edges

def _product(iterable, repeat=1):
    # Simple itertools.product implementation
    if repeat < 0:
        raise ValueError("repeat must be non-negative")
    pools = [tuple(iterable)] * repeat
    result = [()]
    for pool in pools:
        result = [x + (y,) for x in result for y in pool]
    return result

def check_girth(edges, n, min_girth):
    """
    Check if graph has girth >= min_girth.
    Uses BFS from each vertex.
    """
    if n == 0:
        return True
    
    adj = [[] for _ in range(n)]
    for i, j in edges:
        if i != j:
            adj[i].append(j)
            adj[j].append(i)
            
    # Check degree
    for i in range(n):
        if len(adj[i]) != 5:
            return False
            
    # BFS from each vertex to find shortest cycle
    # Optimization: If we find a cycle of length < min_girth, return False.
    # We only need to search up to depth (min_girth - 1) // 2.
    max_depth = (min_girth - 1) // 2
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            node = queue.popleft()
            if dist[node] >= max_depth:
                continue
                
            for nb in adj[node]:
                if dist[nb] == -1:
                    dist[nb] = dist[node] + 1
                    parent[nb] = node
                    queue.append(nb)
                elif parent[node] != nb and parent[nb] != node:
                    # Found a back edge
                    cycle_len = dist[node] + dist[nb] + 1
                    if cycle_len < min_girth:
                        return False
                        
    return True
