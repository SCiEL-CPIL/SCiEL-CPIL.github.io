"""Combined program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import deque

_DEADLINE = 120.0
_MARGIN = 5.0

def _validate_graph(n, edges, k, g, max_vertices=None):
    """
    Validate simple k-regular graph with girth >= g.
    Returns True if valid, False otherwise.
    """
    if max_vertices is not None and n > max_vertices:
        return False
        
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        if u == v or u < 0 or v < 0 or u >= n or v >= n:
            return False
        if v in adj[u]:
            return False
        adj[u].add(v)
        adj[v].add(u)
        
    # Check degree
    for i in range(n):
        if len(adj[i]) != k:
            return False
            
    # Check girth using non-backtracking BFS from each vertex
    for s in range(n):
        dq = deque()
        dist = {}
        for nb in adj[s]:
            dq.append((nb, s))
            dist[(nb, s)] = 1
            
        while dq:
            curr, prev = dq.popleft()
            depth = dist[(curr, prev)]
            
            if curr == s:
                if depth + 1 < g:
                    return False
                continue
                
            for nb in adj[curr]:
                if nb == prev:
                    continue
                state = (nb, curr)
                if state not in dist:
                    dist[state] = depth + 1
                    dq.append(state)
                    
    return True

def _check_girth_fast(adj, n, g):
    """
    Fast girth check: return True if girth >= g, False otherwise.
    Uses non-backtracking BFS with early termination.
    """
    for s in range(n):
        dq = deque()
        dist = {}
        for nb in adj[s]:
            dq.append((nb, s))
            dist[(nb, s)] = 1
            
        while dq:
            curr, prev = dq.popleft()
            depth = dist[(curr, prev)]
            
            if curr == s:
                if depth + 1 < g:
                    return False
                continue
                
            for nb in adj[curr]:
                if nb == prev:
                    continue
                state = (nb, curr)
                if state not in dist:
                    dist[state] = depth + 1
                    dq.append(state)
                    
    return True

def _luw_graph(k, g):
    """
    Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.
    Returns (n, edges). Girth is at least 2*d+2.
    """
    q = k
    d = (g - 2) // 2
    if d < 3:
        d = 3
        
    if q == 4:
        table = [[0, 0, 0, 0], 
                 [0, 1, 2, 3], 
                 [0, 2, 3, 1], 
                 [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
        
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        options = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = options[r]

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

def _try_foster_320():
    """
    Try to construct a 4-regular graph on 320 vertices with girth >= 10.
    Uses a voltage graph approach with a small base.
    """
    # Try base = K_4 (4 vertices, 4-regular) with voltage group Z_80
    # K_4 has edges: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    # Each vertex has degree 3 in K_4, not 4. So K_4 is 3-regular.
    # Need a 4-regular base. Use the 4-cube? No, that's 4-regular on 16 vertices.
    # 16 * 20 = 320. So base = 4-cube (16 vertices, 4-regular), voltage = Z_20.
    
    # 4-cube vertices: 0..15, edge between i and j if Hamming distance 1.
    # Let's try this.
    
    n_base = 16
    m = 20
    n_total = n_base * m  # 320
    
    # 4-cube edges
    base_edges = []
    for i in range(n_base):
        for bit in range(4):
            j = i ^ (1 << bit)
            if i < j:
                base_edges.append((i, j))
    
    # Assign voltages to each directed edge
    # We need to find voltage assignments such that the lift has girth >= 10
    # This is a complex constraint satisfaction problem.
    
    # For now, try random voltage assignments and check girth
    rng = random.Random(12345)
    
    for attempt in range(50):
        # Assign random voltages to each directed edge
        voltages = {}
        for u, v in base_edges:
            voltages[(u, v)] = rng.randint(0, m - 1)
            voltages[(v, u)] = (m - voltages[(u, v)]) % m
        
        # Build the lift
        adj = [set() for _ in range(n_total)]
        for u, v in base_edges:
            vu = voltages[(u, v)]
            vv = voltages[(v, u)]
            for i in range(m):
                a = u * m + i
                b = v * m + ((i + vu) % m)
                adj[a].add(b)
                adj[b].add(a)
        
        # Check girth
        if _check_girth_fast(adj, n_total, 10):
            edges = []
            for u in range(n_total):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            return n_total, edges
    
    return None

def _try_smaller_bases():
    """
    Try other base graph sizes for voltage construction.
    """
    # Try base with 8 vertices (cubical graph or other 4-regular)
    # 8 * 40 = 320
    # 10 * 32 = 320
    # 20 * 16 = 320
    
    # For 8 vertices, 4-regular: the complete graph K_8 is 7-regular, not 4.
    # The 8-vertex 4-regular graph: complement of perfect matching? 
    # K_8 minus a perfect matching is 6-regular. 
    # Actually, the 8-vertex 4-regular graph could be the 4-cube's... no.
    # Let's use a different approach.
    
    return None

def construct(k, g):
    t_start = time.time()
    deadline = t_start + _DEADLINE - _MARGIN
    
    # 1. Try LUW Graph as reliable fallback
    n_luw, edges_luw = _luw_graph(k, g)
    best_n = n_luw
    best_edges = edges_luw
    
    # 2. If k=4, g=10, try to find a smaller graph (Foster graph, n=320)
    if k == 4 and g == 10:
        target_n = 320
        
        # Try voltage graph construction
        if time.time() < deadline - 30:
            result = _try_foster_320()
            if result is not None:
                n_try, edges_try = result
                if _validate_graph(n_try, edges_try, k, g):
                    best_n = n_try
                    best_edges = edges_try
                    return best_n, best_edges
        
        # If we still have time, try more random voltage assignments
        if time.time() < deadline - 10:
            rng = random.Random(999)
            n_base = 16
            m = 20
            n_total = n_base * m
            
            base_edges = []
            for i in range(n_base):
                for bit in range(4):
                    j = i ^ (1 << bit)
                    if i < j:
                        base_edges.append((i, j))
            
            for attempt in range(200):
                if time.time() > deadline:
                    break
                    
                voltages = {}
                for u, v in base_edges:
                    voltages[(u, v)] = rng.randint(0, m - 1)
                    voltages[(v, u)] = (m - voltages[(u, v)]) % m
                
                adj = [set() for _ in range(n_total)]
                for u, v in base_edges:
                    vu = voltages[(u, v)]
                    for i in range(m):
                        a = u * m + i
                        b = v * m + ((i + vu) % m)
                        adj[a].add(b)
                        adj[b].add(a)
                
                if _check_girth_fast(adj, n_total, 10):
                    edges = []
                    for u in range(n_total):
                        for v in adj[u]:
                            if u < v:
                                edges.append((u, v))
                    if _validate_graph(n_total, edges, k, g):
                        best_n = n_total
                        best_edges = edges
                        break
                        
    return best_n, best_edges

# EVOLVE-BLOCK-END
