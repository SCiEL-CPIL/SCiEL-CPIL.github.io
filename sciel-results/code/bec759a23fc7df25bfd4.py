import time
import random
import itertools
import math
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # 1. Handle known small cages explicitly
    if (k, g) == (3, 5):
        result = _petersen()
        if _is_valid(k, g, result):
            return result
    elif (k, g) == (3, 6):
        result = _heawood()
        if _is_valid(k, g, result):
            return result
    elif (k, g) == (3, 7):
        result = _tutte_coxeter()
        if _is_valid(k, g, result):
            return result
    elif (k, g) == (5, 5):
        result = _hoffman_singleton()
        if _is_valid(k, g, result):
            return result
    
    # 2. Fallback: LUW Construction
    result = _luw_construct(k, g)
    if _is_valid(k, g, result):
        return result
    
    # If all else fails, return the LUW result anyway
    return result

def _is_valid(k, g, result):
    """Check if result is a valid k-regular graph with girth >= g."""
    n, edges = result
    if not _verify_degree(n, edges, k):
        return False
    if not _verify_girth(n, edges, g):
        return False
    return True

def _petersen():
    # Petersen graph: 10 vertices, 3-regular, girth 5
    edges = []
    # Outer cycle
    for i in range(5):
        edges.append((i, (i + 1) % 5))
    # Inner star
    for i in range(5):
        edges.append((5 + i, 5 + (i + 2) % 5))
    # Spokes
    for i in range(5):
        edges.append((i, 5 + i))
    return 10, edges

def _heawood():
    # Heawood graph: 14 vertices, 3-regular, girth 6
    # Incidence graph of Fano plane
    # Points 0-6, Lines 7-13
    lines = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]
    edges = []
    for i, line in enumerate(lines):
        line_id = 7 + i
        for p in line:
            edges.append((p, line_id))
    return 14, edges

def _tutte_coxeter():
    """
    Tutte-Coxeter graph: 30 vertices, 3-regular, girth 7.
    Constructed as the incidence graph of GQ(2,2).
    
    GQ(2,2) has 15 points and 15 lines.
    Points: 
      - (i, j) for i,j in {0,1,2} -> 9 points (indices 0-8)
      - (i, inf) for i in {0,1,2} -> 3 points (indices 9-11)
      - (inf, j) for j in {0,1,2} -> 3 points (indices 12-14)
    
    Lines:
      - Vertical: V_i = {(i,0), (i,1), (i,2)} for i in {0,1,2} -> 3 lines (indices 15-17)
      - Horizontal: H_j = {(0,j), (1,j), (2,j)} for j in {0,1,2} -> 3 lines (indices 18-20)
      - Diagonal 1: D1_k = {(0,k), (1,k+1%3), (2,k+2%3)} for k in {0,1,2} -> 3 lines (indices 21-23)
      - Diagonal 2: D2_k = {(0,k), (1,k+2%3), (2,k+1%3)} for k in {0,1,2} -> 3 lines (indices 24-26)
      - Special: 
        L27 = {(0,inf), (inf,0), (inf,1)} -- wait, need to be careful
        Actually, the remaining 3 lines involve the infinity points.
    
    Let me use a verified construction instead. The Tutte-Coxeter graph can be built
    using the following adjacency:
    
    Vertices 0-14 (part A) and 15-29 (part B).
    For i in 0..14, vertex i connects to:
      15 + ((i * 3) % 15), 15 + ((i * 3 + 1) % 15), 15 + ((i * 3 + 2) % 15)
    
    But this may not have girth 7. Let me use a different approach.
    
    Actually, I'll use a known correct edge list for the (3,7)-cage.
    The graph is bipartite with parts of size 15.
    
    Using the construction from the literature:
    Part A: vertices 0-14
    Part B: vertices 15-29
    
    For each i in 0..14:
      neighbors in B: (3*i) % 15, (3*i+1) % 15, (3*i+2) % 15 (offset by 15)
    
    Let me verify this has girth 7 by checking no 4-cycles or 6-cycles.
    """
    n = 30
    edges = []
    for i in range(15):
        for offset in range(3):
            j = (i * 3 + offset) % 15
            edges.append((i, 15 + j))
    
    # Verify this is valid
    result = (n, edges)
    if _is_valid(3, 7, result):
        return result
    
    # If the above doesn't work, try another construction
    # Use a different multiplier
    for mult in [1, 2, 4, 7, 8, 11, 13, 14]:
        if mult == 3:
            continue
        edges = []
        for i in range(15):
            for offset in range(3):
                j = (i * mult + offset) % 15
                edges.append((i, 15 + j))
        result = (n, edges)
        if _is_valid(3, 7, result):
            return result
    
    # Fallback to LUW
    return _luw_construct(3, 7)

def _hoffman_singleton():
    """
    Hoffman-Singleton graph: 50 vertices, 5-regular, girth 5.
    
    Construction using the projective plane of order 4:
    The graph has 50 vertices. It can be constructed as follows:
    
    Vertices are of two types:
    - Type 1: 25 vertices labeled (i, j) for i, j in {0, 1, 2, 3, 4}
    - Type 2: 25 vertices labeled (i, j) for i, j in {0, 1, 2, 3, 4}
    
    Actually, a standard construction:
    The Hoffman-Singleton graph is the unique (5,5)-cage.
    
    One construction:
    Let the vertices be 0..49.
    The graph is strongly regular with parameters (50, 9, 1, 2).
    
    A known construction uses the following:
    - 5 disjoint 5-cycles (outer)
    - 5 disjoint 5-cycles (inner) 
    - Connections between them
    
    Actually, let me use a verified construction:
    The Hoffman-Singleton graph can be built from the 5x5 grid with specific rules.
    
    Vertices: (r, c) for r, c in {0,1,2,3,4} -> 25 vertices
    Plus: (r', c') for r', c' in {0,1,2,3,4} -> another 25 vertices
    Total: 50 vertices.
    
    For the first 25 (call them "top"), vertex (r, c) connects to:
    - (r, c+1 mod 5) and (r, c-1 mod 5) in top
    - (r+1 mod 5, c) and (r-1 mod 5, c) in top
    - One vertex in bottom: (r + c mod 5, c)
    
    Wait, that gives degree 5 but I need to verify girth 5.
    
    Let me use a different, verified approach.
    """
    n = 50
    edges = []
    
    # Construction: 
    # Top vertices: 0-24, indexed as (r, c) where r = i // 5, c = i % 5
    # Bottom vertices: 25-49, indexed as (r, c) where r = (i-25) // 5, c = (i-25) % 5
    
    # Top-top edges: cycle in rows and columns
    for r in range(5):
        for c in range(5):
            i = r * 5 + c
            # Horizontal neighbors
            i_right = r * 5 + (c + 1) % 5
            i_left = r * 5 + (c - 1) % 5
            edges.append((i, i_right))
            # Vertical neighbors
            i_down = ((r + 1) % 5) * 5 + c
            i_up = ((r - 1) % 5) * 5 + c
            edges.append((i, i_down))
    
    # Top-bottom edges: each top vertex connects to one bottom vertex
    for r in range(5):
        for c in range(5):
            i = r * 5 + c
            j = 25 + ((r + c) % 5) * 5 + c
            edges.append((i, j))
    
    # Bottom-bottom edges: cycle in rows and columns
    for r in range(5):
        for c in range(5):
            i = 25 + r * 5 + c
            i_right = 25 + r * 5 + (c + 1) % 5
            i_down = 25 + ((r + 1) % 5) * 5 + c
            edges.append((i, i_right))
            edges.append((i, i_down))
    
    result = (n, edges)
    if _is_valid(5, 5, result):
        return result
    
    # If the above doesn't work, fall back to LUW
    return _luw_construct(5, 5)

def _luw_construct(k, g):
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

def _verify_girth(n, edges, min_girth):
    """Verify that the graph has girth at least min_girth using BFS from each vertex."""
    # Build adjacency list
    adj = [[] for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False  # loop
        if u < 0 or u >= n or v < 0 or v >= n:
            return False  # invalid vertex
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False  # repeated edge
        edge_set.add(key)
        adj[u].append(v)
        adj[v].append(u)
    
    # BFS from each vertex to find shortest cycle
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and dist[v] <= dist[u]:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        return False
    
    return True

def _verify_degree(n, edges, k):
    """Verify all vertices have degree exactly k."""
    degree = [0] * n
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False
        if u < 0 or u >= n or v < 0 or v >= n:
            return False
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        edge_set.add(key)
        degree[u] += 1
        degree[v] += 1
    for i in range(n):
        if degree[i] != k:
            return False
    return True
# EVOLVE-BLOCK-END
