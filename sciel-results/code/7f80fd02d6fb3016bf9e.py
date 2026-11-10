# EVOLVE-BLOCK-START
import time
import random
import math

def construct(k, g):
    """
    Construct a k-regular graph of girth at least g.
    Uses a lifting (covering) approach based on the Heawood graph (Klein Quartic).
    For k=3, g=16, we attempt to find a small cover of the Heawood graph.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit

    if k != 3:
        # Fallback for other k values
        return _lnw_construct(k, g)

    # For k=3, g=16, try Heawood covers
    # Heawood graph has 14 vertices, 21 edges, girth 7.
    # A cover of degree m has 14*m vertices.
    # We want girth >= 16.
    
    heawood_edges = _get_heawood_edges()
    n_base = 14
    m_base_edges = 21
    
    best_n = float('inf')
    best_edges = None
    
    # Try covers of degree m
    # The Moore bound is 512, so 14*m < 512 => m < 36.6.
    # Try m from 10 up to 40.
    for m in range(10, 41):
        if time.time() > deadline - 20:
            break
            
        n_cover = 14 * m
        # If n_cover is already larger than our best known (936), stop
        if n_cover > 936:
            break
            
        # Try to find a valid cover
        # We use cyclic covers Z_m for simplicity and efficiency in checking.
        # c_e in Z_m.
        # Constraint: For every cycle C in base, the sum of c_e along C must not be 0 (mod m) for L>=8.
        # For L=7, the sum S must have gcd(S, m) <= m/3.
        
        # We will just generate random c_e and check the actual girth of the cover.
        # This is more robust than checking all cycles explicitly if the graph is small enough.
        
        attempts = 50
        found = False
        for _ in range(attempts):
            if time.time() > deadline - 10:
                break
                
            c = [random.randint(0, m-1) for _ in range(m_base_edges)]
            
            # Build the cover graph
            edges = []
            # Vertex (u, i) where u in 0..13, i in 0..m-1
            # Edge (u, v) in base with index idx connects (u, i) to (v, (i + c[idx]) % m)
            # and (v, j) to (u, (j - c[idx]) % m) ?? No, it's a regular cover.
            # If we orient the base edges, say u < v, then edge idx connects (u, i) to (v, i + c[idx]) mod m.
            # We need to be careful with the direction.
            
            # Let's define the base edges with indices.
            # We need a consistent orientation for the sum check if we were doing it analytically.
            # For construction, we just add edges.
            
            for idx, (u, v) in enumerate(heawood_edges):
                for i in range(m):
                    j = (i + c[idx]) % m
                    # Edge from (u, i) to (v, j)
                    # Vertex ID: u * m + i
                    u_id = u * m + i
                    v_id = v * m + j
                    edges.append((u_id, v_id))
            
            # Check girth
            adj = _edges_to_adj(edges, n_cover)
            girth = _calculate_girth(adj, n_cover)
            
            if girth >= g:
                best_n = n_cover
                best_edges = edges
                found = True
                break
                
        if found:
            # Since n grows with m, the first valid m is the smallest
            return best_n, best_edges

    if best_edges is not None:
        return best_n, best_edges
        
    # Fallback: LUW construction
    # For k=3, d=8 gives n=26244, girth ~ 16.
    # Or we can try a random graph of size 1000 and repair?
    # LUW is safer for correctness.
    return _lnw_construct(3, g)

def _get_heawood_edges():
    # Heawood graph (Klein Quartic)
    # 14 vertices, 3-regular, girth 7.
    # One standard embedding:
    # Vertices 0-13.
    # Edges:
    edges = []
    # Fano plane based construction or explicit list
    # Explicit list of edges for Heawood graph:
    # 0-1, 0-2, 0-3
    # 1-4, 1-5, 1-6
    # 2-7, 2-8, 2-9
    # 3-10, 3-11, 3-12
    # 4-7, 4-8, 4-9  (Wait, 4 is connected to 1, and needs 2 more)
    # Let's use a known list.
    # Source: https://en.wikipedia.org/wiki/Heawood_graph
    edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 4), (1, 5), (1, 6),
        (2, 7), (2, 8), (2, 9),
        (3, 10), (3, 11), (3, 12),
        (4, 7), (4, 8), (4, 9), # 4 connected to 1, 7, 8, 9? No, degree 3.
        # Let's verify degrees.
        # 0: 1,2,3 (deg 3)
        # 1: 0,4,5,6 (deg 4) -> Error.
        
        # Correct Heawood edges:
        # 0: 1, 2, 3
        # 1: 0, 4, 5
        # 2: 0, 6, 7
        # 3: 0, 8, 9
        # 4: 1, 10, 11
        # 5: 1, 12, 13
        # 6: 2, 10, 12
        # 7: 2, 11, 13
        # 8: 3, 10, 13
        # 9: 3, 11, 12
        # 10: 4, 6, 8
        # 11: 4, 7, 9
        # 12: 5, 6, 9
        # 13: 5, 7, 8
        
        # Let's list them uniquely:
        (0, 1), (0, 2), (0, 3),
        (1, 4), (1, 5),
        (2, 6), (2, 7),
        (3, 8), (3, 9),
        (4, 10), (4, 11),
        (5, 12), (5, 13),
        (6, 10), (6, 12),
        (7, 11), (7, 13),
        (8, 10), (8, 13),
        (9, 11), (9, 12)
    ]
    return edges

def _edges_to_adj(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _calculate_girth(adj, n):
    min_cycle = n + 1
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            if dist[u] >= min_cycle:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    if v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
    return min_cycle if min_cycle <= n else n + 1

def _lnw_construct(k, g):
    # Original fallback
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    return _lnw_construct_params(k, d)

def _lnw_construct_params(k, d):
    """
    LUW construction with fixed q=k and variable d.
    """
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    
    # Rule generation depends on d
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
