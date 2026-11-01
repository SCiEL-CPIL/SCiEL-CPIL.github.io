"""Construct a 5-regular graph of girth 9 via direct randomized construction with girth verification."""
import random
import time
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    For k=5, g=9: use randomized construction with girth checking.
    """
    if k != 5 or g != 9:
        return _luw_construct(k, g)
    
    # Try to construct a 5-regular graph with girth >= 9
    # Use random generation with girth verification
    start_time = time.time()
    timeout = 110  # Leave 10s buffer
    
    # Try different sizes, starting from smaller
    # Moore bound for (5,9) is 1 + 5*(1+4+16+64+256) = 1706
    # Best known is 1116, so try sizes around there
    
    for n in [100, 150, 200, 300, 400, 500, 600, 800, 1000, 1116, 1200, 1500, 1706]:
        if time.time() - start_time > timeout:
            break
        
        # Try multiple random graphs of this size
        for attempt in range(50):
            if time.time() - start_time > timeout:
                break
            
            graph = _random_regular_graph(n, 5)
            if graph is None:
                continue
            
            # Check girth
            girth = _compute_girth(n, graph)
            if girth >= g:
                return n, graph
    
    # Fallback: use LUW construction
    return _luw_construct(5, 9)


def _random_regular_graph(n, k):
    """Generate a random k-regular graph on n vertices."""
    if n * k % 2 != 0:
        return None
    
    # Use configuration model
    half_edges = []
    for v in range(n):
        for _ in range(k):
            half_edges.append(v)
    
    random.shuffle(half_edges)
    
    edges = set()
    adjacency = [set() for _ in range(n)]
    
    for i in range(0, len(half_edges), 2):
        u, v = half_edges[i], half_edges[i+1]
        if u == v:
            continue
        if (u, v) in edges or (v, u) in edges:
            continue
        edges.add((min(u,v), max(u,v)))
        adjacency[u].add(v)
        adjacency[v].add(u)
    
    # Check all vertices have degree k
    for v in range(n):
        if len(adjacency[v]) != k:
            return None
    
    # Convert to list of edges
    edge_list = list(edges)
    if len(edge_list) != n * k // 2:
        return None
    
    return edge_list


def _compute_girth(n, edges):
    """Compute the girth of a graph using BFS from each vertex."""
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    min_girth = float('inf')
    
    for start in range(n):
        # BFS from start, tracking parent to avoid backtracking
        visited = {start: (None, None)}  # node -> (parent, depth)
        queue = deque([(start, None, 0)])
        
        while queue:
            node, parent, depth = queue.popleft()
            
            for neighbor in adj[node]:
                if neighbor == parent:
                    continue
                if neighbor in visited:
                    # Found a cycle
                    cycle_len = depth + visited[neighbor][1] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                else:
                    visited[neighbor] = (node, depth + 1)
                    queue.append((neighbor, node, depth + 1))
        
        # Early termination if girth found
        if min_girth <= 3:
            break
    
    return min_girth if min_girth != float('inf') else n  # If no cycle, return n


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph construction."""
    q = k
    # For girth g, we need d such that the LUW graph has girth at least g.
    # The LUW graph CD(d, q) has girth at least d + 5 for odd d.
    # So we need d + 5 >= g, i.e., d >= g - 5.
    # For odd d: d = g - 5 if (g - 5) is odd, else g - 4.
    
    d = g - 5 if (g - 5) % 2 == 1 else g - 4
    if d < 3:
        d = 3
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    # Define the rule for each coordinate
    rule = {}
    # For c = 2, 3, 4, use specific rules
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)
    
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
