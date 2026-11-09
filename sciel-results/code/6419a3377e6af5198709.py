import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin before 120s limit
    
    # Calculate Moore Bound
    if g % 2 == 1:
        r = (g - 1) // 2
        lb = 1 + k * sum((k-1)**i for i in range(r))
    else:
        r = g // 2
        lb = 2 * sum((k-1)**i for i in range(r-1))
        
    lb = max(lb, 1)
    
    # Try LUW construction first (fast, known to work)
    luw_result = _luw_construct(k, g)
    if luw_result is not None:
        n_luw, edges_luw = luw_result
        # Quick verification
        if _verify_girth_fast(edges_luw, n_luw, k, g):
            # Try to find something better with remaining time
            remaining = deadline - time.time()
            if remaining > 10.0:
                better = _greedy_search(k, g, n_luw, deadline)
                if better is not None:
                    return better
            return luw_result
    
    # Fallback: try random construction
    return _random_construct(k, g, deadline)

def _luw_construct(k, g):
    """Construct a LUW graph."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 == 0 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return x * y % q
    
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
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges

def _verify_girth_fast(edges, n, k, g):
    """Quick girth verification using BFS from each vertex."""
    if n < 1:
        return False
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        if u >= n or v >= n or u == v:
            return False
        adj[u].add(v)
        adj[v].add(u)
    
    # Check degree
    for i in range(n):
        if len(adj[i]) != k:
            return False
    
    # Check for short cycles using BFS
    half_g = (g - 1) // 2
    for start in range(n):
        # BFS from start, tracking distance and parent
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            node = queue.popleft()
            if dist[node] >= half_g:
                continue
            for neighbor in adj[node]:
                if dist[neighbor] == -1:
                    dist[neighbor] = dist[node] + 1
                    parent[neighbor] = node
                    queue.append(neighbor)
                elif neighbor != parent[node]:
                    # Found a cycle
                    cycle_len = dist[node] + dist[neighbor] + 1
                    if cycle_len < g:
                        return False
    
    return True

def _greedy_search(k, g, max_n, deadline):
    """Try greedy construction to find a smaller graph."""
    # This is a placeholder - the greedy backtracking approach
    # would be very complex to implement correctly in limited time
    # For now, return None to use the LUW result
    return None

def _random_construct(k, g, deadline):
    """Random construction as last resort."""
    start_time = time.time()
    lb = 1
    if g % 2 == 1:
        r = (g - 1) // 2
        lb = 1 + k * sum((k-1)**i for i in range(r))
    else:
        r = g // 2
        lb = 2 * sum((k-1)**i for i in range(r-1))
    lb = max(lb, 1)
    
    for n in range(lb, lb + 200):
        if time.time() > deadline - 5.0:
            break
        if n * k % 2 != 0:
            continue
        
        # Try a few random regular graphs
        for attempt in range(20):
            if time.time() > deadline - 5.0:
                break
            graph = _gen_random_regular(n, k)
            if graph is None:
                continue
            edges = _graph_to_edges(graph)
            if _verify_girth_fast(edges, n, k, g):
                return n, edges
    
    return None

def _gen_random_regular(n, k):
    if n * k % 2 != 0 or k >= n:
        return None
    half_edges = []
    for i in range(n):
        for _ in range(k):
            half_edges.append(i)
    random.shuffle(half_edges)
    
    graph = [set() for _ in range(n)]
    for i in range(0, len(half_edges), 2):
        u, v = half_edges[i], half_edges[i+1]
        if u == v or v in graph[u]:
            return None
        graph[u].add(v)
        graph[v].add(u)
    return graph

def _graph_to_edges(graph):
    edges = []
    for i in range(len(graph)):
        for j in graph[i]:
            if i < j:
                edges.append((i, j))
    return edges
