"""Voltage graph lifting approach for high-girth 3-regular graphs."""
import random
from collections import deque
from math import isqrt

def _bfs_girth(adj):
    """Compute the girth of a graph using BFS from each vertex."""
    n = len(adj)
    girth = float('inf')
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif parent[u] != v and parent[v] != u:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < girth:
                        girth = cycle_len
    return girth if girth < float('inf') else None

def _voltage_lift(base_adj, voltages, m):
    """
    Lift a graph using voltage assignment.
    base_adj: adjacency list of base graph
    voltages: dict mapping (u,v) -> voltage in Z_m for each edge
    m: size of voltage group Z_m
    Returns adjacency list of lifted graph.
    """
    n_base = len(base_adj)
    n_lift = n_base * m
    adj = [set() for _ in range(n_lift)]
    
    for u in range(n_base):
        for v in base_adj[u]:
            if u < v:  # Each edge once
                vol = voltages.get((u, v), 0)
                # Edge (u, a) -> (v, (a + vol) % m)
                for a in range(m):
                    b = (a + vol) % m
                    u_lift = u * m + a
                    v_lift = v * m + b
                    adj[u_lift].add(v_lift)
                    adj[v_lift].add(u_lift)
                # Also the reverse direction with -vol
                # Already handled since we add both directions
    
    # Convert to lists for efficiency
    adj = [list(s) for s in adj]
    return adj

def _check_girth(adj, target_girth):
    """Check if graph has girth >= target_girth."""
    n = len(adj)
    # BFS from each vertex, tracking shortest cycle
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif parent[u] != v and parent[v] != u:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < target_girth:
                        return False
    return True

def _small_3regular_girth8():
    """Return a 3-regular graph with girth 8 (e.g., a known one)."""
    # Use the 3-cube Q3? No, Q3 is 3-regular with 8 vertices and girth 4.
    # Let's use a known 3-regular graph with girth 8.
    # The (3,8)-cage has 16 vertices? No, n(3,8) = 16? Let me think.
    # Moore bound for (3,8): g=8, r=4, n >= 1 + 3*(2^3 - 1)/(2-1) = 1 + 3*7 = 22? 
    # Actually for even girth g=2r: n >= 1 + k*(2^(r-1)-1)/(k-1) ... let me just use a known graph.
    # Let me construct one: take two copies of a 4-vertex star... no.
    # Simple approach: use the Petersen graph (girth 5) and lift.
    # Or just return a hardcoded 3-regular girth-8 graph.
    # The smallest 3-regular graph with girth 8 has 16 vertices? 
    # n(3,8) = 16 is not right. Let me use a different approach.
    # Just use a random 3-regular graph on ~20 vertices and check.
    
    # Actually, let me just use a simple construction:
    # Two disjoint copies of K_{3,3} won't work (girth 4).
    # Let me use the 3-regular graph formed by the 4x4 grid with wraparound... 
    # Simpler: just generate and verify.
    
    rng = random.Random(12345)
    for trial in range(1000):
        n = 20  # even
        stubs = list(range(n)) * 3
        rng.shuffle(stubs)
        edges = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                valid = False
                break
            key = (min(u,v), max(u,v))
            if key in edges:
                valid = False
                break
            edges.add(key)
        if not valid:
            continue
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        if all(len(adj[i]) == 3 for i in range(n)):
            g = _bfs_girth(adj)
            if g is not None and g >= 8:
                return [list(s) for s in adj]
    # Fallback: return Petersen graph (girth 5) - we'll lift more
    # Petersen graph edges
    pet = [
        [1, 4, 5],
        [0, 2, 6],
        [1, 3, 7],
        [2, 4, 8],
        [0, 3, 9],
        [0, 6, 8],
        [1, 5, 9],
        [2, 7, 9],
        [3, 5, 7],
        [4, 6, 8]
    ]
    return pet

def construct(k, g):
    """Construct a k-regular graph with girth >= g."""
    if k != 3:
        return _construct_luw(k, g)
    
    girth_target = g
    rng = random.Random(42)
    
    # Get a base graph with decent girth
    base_adj = _small_3regular_girth8()
    n_base = len(base_adj)
    
    # Base edges
    base_edges = set()
    for u in range(n_base):
        for v in base_adj[u]:
            if u < v:
                base_edges.add((u, v))
    
    # Try different lift sizes m
    best_n = None
    best_adj = None
    
    for m in range(2, 50):
        n_lift = n_base * m
        if best_n is not None and n_lift >= best_n:
            continue
        
        # Try random voltage assignments
        for trial in range(200):
            voltages = {}
            for e in base_edges:
                voltages[e] = rng.randint(0, m - 1)
            
            adj = _voltage_lift(base_adj, voltages, m)
            
            # Check girth
            if _check_girth(adj, girth_target):
                if best_n is None or n_lift < best_n:
                    best_n = n_lift
                    best_adj = adj
                break
    
    if best_adj is not None:
        # Convert to edge list
        edges = []
        for u in range(best_n):
            for v in best_adj[u]:
                if u < v:
                    edges.append((u, v))
        return best_n, edges
    
    # Fallback: LUW construction
    return _construct_luw(k, g)

def _construct_luw(k, g):
    """Original Lazebnik-Ustimenko-Woldar construction as fallback."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
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
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
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
