import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0

    # Handle small known cases for k=3
    if k == 3:
        if g <= 3:
            return 4, [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
        elif g <= 4:
            return 6, [(0,3),(0,4),(0,5),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5)]
        elif g <= 5:
            edges = []
            for i in range(5):
                edges.append((i, (i+1)%5))
                edges.append((5+i, 5+(i+2)%5))
                edges.append((i, 5+i))
            return 10, edges
        elif g <= 6:
            # Heawood graph
            edges = []
            fano_lines = [(0,1,2),(0,3,4),(0,5,6),(1,3,5),(1,4,6),(2,3,6),(2,4,5)]
            for i, line in enumerate(fano_lines):
                for p in line:
                    edges.append((p, 7+i))
            return 14, edges
        elif g <= 7:
            # (3,7)-cage is 24 vertices (Foster graph is (3,7)? No, Foster is (3,7) with 24 verts? 
            # Actually n(3,7)=24. The unique (3,7)-cage is the Foster graph? 
            # Wait, the (3,7)-cage is the unique cubic graph of girth 7 on 24 vertices.
            # It is not the Foster graph (which is (3,7) but often cited as 24).
            # Let's use a known construction or LUW.
            pass

    if k == 5 and g == 9:
        # Try to construct the 1116-vertex cage via voltage cover search
        n, edges = _construct_1116_cage()
        if n == 1116 and _verify_girth_fast(n, edges, 9):
            return n, edges
        
        # Fallback to LUW construction
        if time.time() < deadline - 15:
            # LUW for k=5, g=9 requires d=4 (girth >= 10)
            n, edges = _luw_construct(5, 4)
            if _verify_girth_fast(n, edges, 9):
                return n, edges
            # Try d=5 just in case (girth >= 11, n=2500)
            n, edges = _luw_construct(5, 5)
            if _verify_girth_fast(n, edges, 9):
                return n, edges

    # General case: LUW construction
    min_d = max(3, g - 5)
    if (g - 5) % 2 != 0:
        min_d = max(3, g - 4)

    for d in range(min_d, min_d + 8):
        if time.time() > deadline - 10:
            break
        n, edges = _luw_construct(k, d)
        if _verify_girth_fast(n, edges, g):
            return n, edges

    d_fallback = min_d + 8
    return _luw_construct(k, d_fallback)

def _construct_1116_cage():
    """
    Constructs the 1116-vertex (5,9)-cage using a voltage cover of K_6 over Z_186.
    Uses local search to find valid voltages.
    """
    n_base = 6
    m = 186
    n_total = n_base * m

    k6_edges = []
    for i in range(n_base):
        for j in range(i+1, n_base):
            k6_edges.append((i, j))
    
    # Map edge pair to index
    edge_idx = {}
    for idx, (u, v) in enumerate(k6_edges):
        edge_idx[(u, v)] = idx
        edge_idx[(v, u)] = idx

    # Generate all closed non-backtracking walks of length 3 to 8
    # A walk is a sequence of vertices v0...vL with v0=vL, vi!=vi-1, vi!=vi+1
    # We represent a walk by the sequence of edge indices (with signs for direction)
    # Since K6 is small, we can enumerate them.
    
    walks = []
    # Precompute neighbors
    neighbors = {i: [j for j in range(n_base) if j != i] for i in range(n_base)}
    
    def gen_walks(start, prev, curr, path_edges, length, max_len):
        if length == 0:
            if curr == start and len(path_edges) >= 3:
                walks.append(list(path_edges))
            return
        if length >= max_len:
            return
        
        for next_v in neighbors[curr]:
            if next_v == prev:
                continue
            # Add edge (curr, next_v)
            e_idx = edge_idx[(curr, next_v)]
            # Direction: +1 if curr < next_v, -1 otherwise? 
            # Voltage addition: if we traverse (u,v) with voltage v_uv, we add v_uv.
            # If we traverse (v,u), we add -v_uv.
            # Let's store (edge_idx, sign)
            sign = 1 if curr < next_v else -1
            path_edges.append((e_idx, sign))
            gen_walks(start, curr, next_v, path_edges, length + 1, max_len)
            path_edges.pop()

    for s in range(n_base):
        for first in neighbors[s]:
            gen_walks(s, s, first, [(edge_idx[(s, first)], 1 if s < first else -1)], 1, 8)

    # Filter walks: ensure non-backtracking (already done by prev check) and length >= 3
    # Note: gen_walks starts with length 1, goes up to 8.
    # We need closed walks.
    
    # Optimization: Remove duplicate walks (same edge sequence)
    # Actually, different vertex sequences can yield same edge sequences? No, edge sequence defines vertex sequence in K6.
    # But we might have symmetries.
    
    # Local Search
    # Objective: Minimize number of walks with sum(voltage) == 0
    
    def count_bad(walks, voltages):
        bad = 0
        for w in walks:
            s = 0
            for e_idx, sign in w:
                s += sign * voltages[e_idx]
            if s % m == 0:
                bad += 1
        return bad

    rng = random.Random(42)
    best_voltages = None
    best_bad = float('inf')
    
    # Initialize with random voltages
    current_voltages = [rng.randint(1, m-1) for _ in range(len(k6_edges))]
    current_bad = count_bad(walks, current_voltages)
    
    # Simulated Annealing / Local Search
    T = 1.0
    for iter in range(20000):
        if time.time() > deadline - 20:
            break
        if current_bad == 0:
            best_voltages = current_voltages[:]
            break
            
        # Pick a random edge to change
        e_idx = rng.randint(0, len(k6_edges) - 1)
        old_val = current_voltages[e_idx]
        new_val = rng.randint(1, m-1)
        current_voltages[e_idx] = new_val
        
        new_bad = count_bad(walks, current_voltages)
        
        # Accept if better or equal (to escape plateaus)
        # Or with probability exp(-(new_bad - current_bad)/T)
        if new_bad <= current_bad:
            current_bad = new_bad
            if new_bad < best_bad:
                best_bad = new_bad
                best_voltages = current_voltages[:]
        else:
            # Revert
            current_voltages[e_idx] = old_val
        
        # Cool down
        T *= 0.999
        
        if T < 0.01:
            T = 0.01

    if best_voltages is None:
        return 0, []

    # Build the graph
    voltages = best_voltages
    edges = []
    seen_edges = set()
    for v in range(n_base):
        for w in range(n_base):
            if v == w:
                continue
            e_idx = edge_idx[(v, w)]
            vol = voltages[e_idx]
            # Direction: if v < w, voltage is vol. If v > w, voltage is -vol.
            # In gen_walks, we used sign based on v<w.
            # Here, we traverse from v to w.
            # If v < w, add vol.
            # If v > w, add -vol.
            if v < w:
                delta = vol
            else:
                delta = -vol
                
            for g_idx in range(m):
                u_idx = v * m + g_idx
                g2 = (g_idx + delta) % m
                w_idx = w * m + g2
                if u_idx < w_idx:
                    edge_key = (u_idx, w_idx)
                else:
                    edge_key = (w_idx, u_idx)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    edges.append(edge_key)

    return n_total, edges

def _verify_girth_fast(n, edges, g):
    if n <= 0:
        return False

    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    if n > 0:
        expected_deg = len(adj[0])
        for i in range(n):
            if len(adj[i]) != expected_deg:
                return False

    max_depth = g // 2

    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])

        while queue:
            u = queue.popleft()
            if dist[u] >= max_depth:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    if dist[v] < max_depth:
                        queue.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False

    return True

def _luw_construct(k, d):
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q

    rule = {}
    rule[2] = (1, 1)
    if d >= 3:
        rule[3] = (2, 1)
    if d >= 4:
        rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
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

    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges
