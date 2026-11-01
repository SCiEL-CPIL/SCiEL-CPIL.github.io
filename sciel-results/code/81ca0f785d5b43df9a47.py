"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

def construct(k, g):
    deadline = time.time() + 140  # safety margin
    # For k=4, g=10: construct 4-regular graph with girth >= 10 on 320 vertices
    # Use a voltage graph / lift construction
    
    if k != 4 or g != 10:
        # Fallback: use LUW construction for other parameters
        return _luw_construct(k, g)
    
    n = 320
    # Construct a 4-regular graph on 320 vertices
    # Use a specific construction: bipartite with parts of 160
    # Edges defined by 4 permutations
    
    # Try construction: vertices (a,b) with a in Z_5, b in Z_64
    # Edges: (a,b)-(a,b+1), (a,b)-(a,b-1), (a,b)-(a+1,b+32), (a,b)-(a-1,b+32)
    # But this has 4-cycles. Instead use a different offset.
    
    # Use a known (4,10)-cage construction via specific edge set
    edges = []
    seen = set()
    
    def add_edge(u, v):
        if u == v:
            return
        a, b = min(u, v), max(u, v)
        if (a, b) not in seen:
            seen.add((a, b))
            edges.append((a, b))
    
    # Construction: use Z_320 with specific connection set
    # Try: i ~ i+1, i ~ i-1, i ~ i+40, i ~ i-40 (has 4-cycles)
    # Try: i ~ i+1, i ~ i-1, i ~ i+80, i ~ i-80 (has 4-cycles)
    
    # Use a non-circulant: bipartite construction
    # Part U: 0..159, Part V: 160..319
    # For u in U, connect to: (u*1) % 160 + 160, (u+1) % 160 + 160, (u+2) % 160 + 160, (u+3) % 160 + 160
    
    # This likely has short cycles. Let me use a more sophisticated construction.
    
    # Use the following: a 4-regular bipartite graph based on a (160, 4, 5) design or similar
    
    # Actually, let me use a random approach with verification
    import random
    random.seed(42)
    
    best_n = None
    best_edges = None
    
    # Try several constructions
    for trial in range(50):
        if time.time() > deadline - 5:
            break
        
        # Construction: 4-regular bipartite graph on 320 vertices
        # U = 0..159, V = 160..319
        # For each u in U, choose 4 neighbors in V such that the resulting graph has high girth
        
        # Use a specific algebraic construction
        # Vertices: (i, j) with i in Z_4, j in Z_80
        # Edges: (i,j) ~ (i, j+1), (i,j) ~ (i, j-1), (i,j) ~ (i+1, j+40), (i,j) ~ (i+1, j-40)
        # mod (4, 80)
        
        n_vertices = 320
        adj = [[] for _ in range(n_vertices)]
        
        def vid(i, j):
            return (i % 4) * 80 + (j % 80)
        
        for i in range(4):
            for j in range(80):
                u = vid(i, j)
                v1 = vid(i, j + 1)
                v2 = vid(i, j - 1)
                v3 = vid(i + 1, j + 40)
                v4 = vid(i + 1, j - 40)
                for v in [v1, v2, v3, v4]:
                    if v != u and v not in adj[u]:
                        adj[u].append(v)
                        adj[v].append(u)
        
        # Check degree
        if all(len(adj[i]) == 4 for i in range(n_vertices)):
            # Check girth
            girth = _check_girth(adj, n_vertices, g)
            if girth >= g:
                best_n = n_vertices
                best_edges = []
                seen_edges = set()
                for u in range(n_vertices):
                    for v in adj[u]:
                        if u < v:
                            if (u, v) not in seen_edges:
                                seen_edges.add((u, v))
                                best_edges.append((u, v))
                break
    
    if best_n is not None:
        return best_n, best_edges
    
    # Fallback: LUW construction
    return _luw_construct(k, g)

def _check_girth(adj, n, min_girth):
    """Check if graph has girth >= min_girth using BFS from each vertex."""
    for start in range(n):
        # BFS to find shortest cycle through start
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
                elif v != parent[u]:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        return cycle_len
    return min_girth  # girth >= min_girth

def _luw_construct(k, g):
    """Original LUW construction as fallback."""
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
# EVOLVE-BLOCK-END
