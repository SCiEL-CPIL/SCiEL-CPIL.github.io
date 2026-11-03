import time
import random
from itertools import combinations

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g on as few vertices as possible.
    Returns (n, edges).
    """
    start_time = time.time()
    deadline = start_time + 115  # safety margin

    # Strategy:
    # 1. Try to find a small k-regular graph with girth g//2 (if g is even) 
    #    and take its bipartite double cover.
    #    For k=4, g=10, we need a 4-regular graph with girth 5.
    #    The smallest 4-regular graph with girth 5 has 16 vertices.
    #    Its double cover has 32 vertices and girth 10.
    # 2. If that fails, fallback to LUW construction (4096 vertices).
    
    if k == 4 and g == 10:
        # Try to construct a 4-regular graph with girth 5 on 16 vertices
        target_n = 16
        target_girth = 5
        
        # Backtracking search for 4-regular graph with girth 5 on 16 vertices
        # This is a small search space.
        # We can use a randomized greedy approach or backtracking.
        # Given the small size, a simple backtracking or even exhaustive search 
        # with pruning is feasible.
        
        # However, generating all 4-regular graphs on 16 vertices is hard.
        # Instead, we can try to construct one explicitly or via random restarts.
        
        # Known graph: The 4-regular (4,5)-cage is the 16-vertex graph.
        # It is the incidence graph of the 2-(16, 6, 2) design? No.
        # It is the 4-dimensional cube Q4 with 4-cycles broken?
        
        # Let's try a random search for a 4-regular graph with girth 5 on 16 vertices.
        # If found, double cover it.
        
        found_graph = None
        attempts = 0
        while time.time() < deadline - 10 and attempts < 5000:
            attempts += 1
            edges = _gen_random_4reg(16)
            if edges is None:
                continue
            if _has_girth_at_least(edges, 16, 5):
                found_graph = edges
                break
        
        if found_graph is not None:
            # Double cover
            n_base = 16
            n_double = 32
            edges_double = []
            for u, v in found_graph:
                # (u, 0) -- (v, 1)
                edges_double.append((u, n_base + v))
                # (v, 0) -- (u, 1)
                edges_double.append((v, n_base + u))
            
            # Verify
            if _has_girth_at_least(edges_double, n_double, 10):
                return n_double, edges_double
        
        # Fallback to LUW
        return _luw_construct(k, g)
    
    # General fallback
    return _luw_construct(k, g)


def _gen_random_4reg(n):
    """Generate a random 4-regular graph on n vertices."""
    if n < 4:
        return None
    if n * 4 % 2 != 0:
        return None
    
    # Use configuration model
    for attempt in range(20):
        stubs = []
        for v in range(n):
            for _ in range(4):
                stubs.append(v)
        
        random.shuffle(stubs)
        
        edges = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                valid = False
                break
            edge = (min(u,v), max(u,v))
            if edge in edges:
                valid = False
                break
            edges.add(edge)
        
        if valid:
            return list(edges)
    
    return None


def _has_girth_at_least(edges, n, g):
    """Check if graph has girth at least g using BFS."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check for cycles of length < g
    # For small n, BFS from each vertex is fast.
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        
        for node in queue:
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


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph construction."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
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
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges
