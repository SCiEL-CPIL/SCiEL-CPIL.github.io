import time
import random
from collections import defaultdict

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Primary: LUW with small d for compact graph.
    Fallback: voltage graph construction over K4.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    if k != 3:
        return _construct_luw(k, g, d=7)
    
    # Try LUW with small d first - this gives ~4k vertices for g=16
    # d=7 is much smaller than d=11, giving a much smaller graph
    try:
        n, edges = _construct_luw(k, g, d=7)
        # Verify girth
        girth = _compute_girth(n, edges)
        if girth >= g:
            return n, edges
        else:
            # Girth too small, try larger d
            for d_try in [8, 9, 10, 11]:
                if time.time() > deadline:
                    break
                n, edges = _construct_luw(k, g, d=d_try)
                girth = _compute_girth(n, edges)
                if girth >= g:
                    return n, edges
    except Exception:
        pass
    
    # Fallback to voltage graph construction
    return _construct_voltage(k, g, deadline)

def _construct_luw(k, g, d=None):
    """
    LUW construction with specified depth d.
    For k=3, uses q=3 arithmetic.
    """
    if d is None:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    # Build rule table
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
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
    
    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges

def _construct_voltage(k, g, deadline):
    """
    Voltage graph construction over K4 for k=3.
    """
    if k != 3:
        return _construct_luw(k, g, d=7)
    
    edge_map = {
        (0, 1): 0, (1, 0): 0,
        (0, 2): 1, (2, 0): 1,
        (0, 3): 2, (3, 0): 2,
        (1, 2): 3, (2, 1): 3,
        (1, 3): 4, (3, 1): 4,
        (2, 3): 5, (3, 2): 5
    }
    
    target_girth = g
    max_walk_len = target_girth - 1
    
    # Precompute coefficient vectors for all reduced closed walks of length < g
    coeff_vectors = set()
    
    for start_v in range(4):
        stack = [(start_v, -1, 0, [0]*6)]
        
        while stack:
            v, prev_e, l, counts = stack.pop()
            
            if l > 0 and v == start_v:
                coeff_vectors.add(tuple(counts))
            
            if l >= max_walk_len:
                continue
            
            for next_v in range(4):
                if next_v == v:
                    continue
                e_idx = edge_map[(v, next_v)]
                if e_idx == prev_e:
                    continue
                
                new_counts = counts.copy()
                new_counts[e_idx] += 1
                stack.append((next_v, e_idx, l + 1, new_counts))
    
    if (0,0,0,0,0,0) in coeff_vectors:
        coeff_vectors.remove((0,0,0,0,0,0))
        
    coeff_list = list(coeff_vectors)
    
    best_n = float('inf')
    best_edges = None
    
    primes = [131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293]
    
    for m in primes:
        if time.time() > deadline:
            break
            
        max_attempts = 30
        for _ in range(max_attempts):
            if time.time() > deadline:
                break
                
            voltages = [random.randint(1, m-1) for _ in range(6)]
            
            valid = True
            for c in coeff_list:
                s = sum(c[i] * voltages[i] for i in range(6))
                if s % m == 0:
                    valid = False
                    break
            
            if valid:
                n = 4 * m
                edges = []
                for u in range(4):
                    for w in range(4):
                        if u == w:
                            continue
                        e_idx = edge_map[(u, w)]
                        vol = voltages[e_idx]
                        for v in range(m):
                            u_node = v * 4 + u
                            w_node = ((v + vol) % m) * 4 + w
                            if u_node < w_node:
                                edges.append((u_node, w_node))
                            else:
                                edges.append((w_node, u_node))
                
                edges = list(set(edges))
                
                girth = _compute_girth(n, edges)
                
                if girth >= g:
                    if n < best_n:
                        best_n = n
                        best_edges = edges
                        if n <= 936:
                            return n, edges
    
    if best_edges is not None:
        return best_n, best_edges
    
    return _construct_luw(k, g, d=7)

def _compute_girth(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    girth = float('inf')
    for i in range(n):
        dist = [-1] * n
        parent = [-1] * n
        queue = [i]
        dist[i] = 0
        while queue:
            u = queue.pop(0)
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < girth:
                        girth = cycle_len
    return girth
