"""Construct a 3-regular graph of girth 16 via voltage graph / circulant lift."""
# EVOLVE-BLOCK-START
import random
from collections import deque

def construct(k, g):
    # For k=3, g=16: use a voltage graph construction to lift a small base graph
    # to a 3-regular graph with girth at least 16.
    
    if k != 3:
        return _luw_construct(k, g)
    
    # Base graph: Heawood graph (14 vertices, 3-regular, girth 6)
    # Vertices 0..13
    # Edges:
    # (0,1), (0,2), (0,3)
    # (1,4), (1,5), (1,6)
    # (2,7), (2,8), (2,9)
    # (3,10), (3,11), (3,12)
    # (4,5), (4,13), (4,11) -- Wait, let's get the exact edges of Heawood.
    # Heawood graph is the bipartite incidence graph of Fano plane.
    # Vertices: 0-6 (points), 7-13 (lines)
    # Point i is connected to Line j if i is on line j.
    # Fano lines:
    # 0: {0,1,2}
    # 1: {0,3,4}
    # 2: {0,5,6}
    # 3: {1,3,5}
    # 4: {1,4,6}
    # 5: {2,3,6}
    # 6: {2,4,5}
    
    heawood_edges = []
    lines = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    for j, line in enumerate(lines):
        line_vertex = 7 + j
        for i in line:
            heawood_edges.append((i, line_vertex))
            
    n_base = 14
    base_adj = [[] for _ in range(n_base)]
    for u, v in heawood_edges:
        base_adj[u].append(v)
        base_adj[v].append(u)
        
    # Precompute non-backtracking walks of length 6 to 15
    # A walk is a sequence of vertices (v0, ..., vl) such that (vi, vi+1) is an edge and vi-1 != vi+1.
    # We need to check if sum of voltages is 0.
    # Since we don't know voltages yet, we just store the list of edges in the walk.
    # Actually, we can just do a DFS for each start vertex and length.
    
    def get_non_backtracking_walks(n, adj, min_len, max_len):
        walks = []
        for start in range(n):
            # Stack: (current, prev, path_edges)
            stack = [(start, -1, [])]
            while stack:
                curr, prev, path = stack.pop()
                l = len(path)
                if min_len <= l <= max_len:
                    walks.append(path)
                if l == max_len:
                    continue
                for nxt in adj[curr]:
                    if nxt != prev:
                        stack.append((nxt, curr, path + [(curr, nxt)]))
        return walks

    # We only need to check walks of length 6 to g-1
    # For g=16, check 6..15
    walks = get_non_backtracking_walks(n_base, base_adj, 6, g-1)
    # There might be many walks. 
    # Heawood has 21 edges. 
    # Number of walks grows as 2^l. 
    # For l=15, approx 2^15 = 32768 walks per start vertex? No, branching factor is 2 (since degree 3, one back).
    # Total walks approx 14 * 2^15 = 458k. This is manageable.
    
    # Try different m
    # Target n is around 900. n = 14 * m. So m ~ 64.
    # Let's try m from 50 to 80.
    
    for m in range(50, 81):
        # Try a few random assignments
        for attempt in range(100):
            random.seed(attempt * 1000 + m)
            # Assign random voltages to edges
            # We need to map edge to voltage
            # Edges in heawood_edges are (u, v). We need a consistent direction for voltage.
            # Let's say voltage is from u to v.
            # If we traverse v to u, voltage is -v_e mod m.
            
            edge_volts = {}
            for u, v in heawood_edges:
                edge_volts[(u, v)] = random.randint(0, m-1)
                edge_volts[(v, u)] = (-edge_volts[(u, v)]) % m
                
            # Check all walks
            valid = True
            for path in walks:
                s = 0
                for u, v in path:
                    s = (s + edge_volts[(u, v)]) % m
                if s == 0:
                    valid = False
                    break
            
            if valid:
                # Construct the lift
                n_lift = n_base * m
                edges = []
                # Vertex (i, alpha) is i * m + alpha
                for u, v in heawood_edges:
                    vol = edge_volts[(u, v)]
                    for alpha in range(m):
                        u_lift = u * m + alpha
                        v_lift = v * m + (alpha + vol) % m
                        edges.append((u_lift, v_lift))
                return n_lift, edges
                
    # Fallback to LUW if voltage construction fails
    return _luw_construct(k, g)

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph construction."""
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
