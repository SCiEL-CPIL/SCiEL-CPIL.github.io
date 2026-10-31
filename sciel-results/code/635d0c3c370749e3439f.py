import time
import random
import numpy as np
from collections import deque

def _build_adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _check_girth_fast(n, edges, min_girth):
    """Fast girth check using BFS from each vertex with parent tracking."""
    adj = _build_adj(n, edges)
    max_depth = (min_girth - 1) // 2

    for start in range(n):
        visited = [-1] * n
        visited[start] = 0
        queue = deque([(start, 0, -1)])

        while queue:
            node, depth, parent = queue.popleft()
            if depth > max_depth:
                continue
            for neighbor in adj[node]:
                if neighbor == parent:
                    continue
                if visited[neighbor] == -1:
                    visited[neighbor] = depth + 1
                    queue.append((neighbor, depth + 1, node))
                else:
                    cycle_len = depth + visited[neighbor] + 1
                    if cycle_len < min_girth:
                        return False
    return True

def _luw_construct(k, g):
    """
    Lazebnik-Ustimenko-Woldar construction for 3-regular graphs.
    We try smaller d values to get fewer vertices, checking girth empirically.
    """
    if k != 3:
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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

    q = 3

    def build_luw(d):
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

    for d in [8, 9, 10, 11]:
        n, edges = build_luw(d)
        if _check_girth_fast(n, edges, g):
            return n, edges
    return build_luw(11)

def _cayley_graph_zm_x_zn(m, n, deadline):
    """
    Construct Cayley graph of Z_m x Z_n with generators:
    S = {(1,1), (-1,-1), (m//2, 0)}
    """
    if m % 2 != 0:
        return None
    
    t = (m // 2, 0)
    s = (1, 1)
    s_inv = ((-1) % m, (-1) % n)
    
    if s == s_inv or s == t or s_inv == t:
        return None
    
    generators = [s, s_inv, t]
    
    n_vertices = m * n
    edges = set()
    
    for i in range(m):
        for j in range(n):
            v = i * n + j
            for (di, dj) in generators:
                ni = (i + di) % m
                nj = (j + dj) % n
                u = ni * n + nj
                if v < u:
                    edges.add((v, u))
    
    if len(edges) != n_vertices * 3 // 2:
        return None
    
    return n_vertices, list(edges)

def _cayley_search(k, g, deadline):
    """
    Search for small 3-regular Cayley graphs with girth >= g.
    Use Z_m x Z_n with the generator scheme above.
    """
    if k != 3:
        return None
    
    best = None
    
    for m in range(6, 200, 2):
        if time.time() > deadline:
            break
        for n in range(3, 200):
            if time.time() > deadline:
                break
            
            result = _cayley_graph_zm_x_zn(m, n, deadline)
            if result is None:
                continue
            
            n_vertices, edges = result
            
            if best is not None and n_vertices >= best[0]:
                continue
            
            if _check_girth_fast(n_vertices, edges, g):
                if best is None or n_vertices < best[0]:
                    best = (n_vertices, edges)
    
    return best

def _voltage_lift_petersen(k, g, deadline, seed=42):
    """
    Voltage lift of the Petersen graph using a deterministic, algebraic voltage
    assignment over a cyclic group of order m.
    """
    if k != 3:
        return None
    
    random.seed(seed)
    
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    n_base = 10
    m_edges = len(petersen_edges)
    
    adj_base = [[] for _ in range(n_base)]
    for i, (u, v) in enumerate(petersen_edges):
        adj_base[u].append((v, i))
        adj_base[v].append((u, i))
    
    cycles = []
    seen = set()
    
    def dfs_cycles(start, curr, visited, edge_path, length):
        for next_v, edge_idx in adj_base[curr]:
            if next_v == start and length >= 3:
                cycle_key = tuple(sorted(edge_path))
                if cycle_key not in seen:
                    seen.add(cycle_key)
                    cycles.append(cycle_key)
            elif next_v not in visited:
                visited.add(next_v)
                dfs_cycles(start, next_v, visited, edge_path + [edge_idx], length + 1)
                visited.remove(next_v)
    
    for start in range(n_base):
        visited = {start}
        dfs_cycles(start, start, visited, [], 0)
    
    primes_to_try = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    for m in primes_to_try:
        if time.time() > deadline:
            break
        
        for a in range(1, m):
            for b in range(m):
                voltages = [(a * i + b) % m for i in range(m_edges)]
                
                valid = True
                for C in cycles:
                    s = sum(voltages[i] for i in C) % m
                    if s == 0:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                n = 10 * m
                edges = []
                for idx, (u, v) in enumerate(petersen_edges):
                    vol = voltages[idx]
                    for j in range(m):
                        a_v = u * m + j
                        b_v = v * m + ((j + vol) % m)
                        edges.append((a_v, b_v))
                
                if _check_girth_fast(n, edges, g):
                    return n, edges
        
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            voltages = [random.randint(1, m - 1) for _ in range(m_edges)]
            
            valid = True
            for C in cycles:
                s = sum(voltages[i] for i in C) % m
                if s == 0:
                    valid = False
                    break
            
            if not valid:
                continue
            
            n = 10 * m
            edges = []
            for idx, (u, v) in enumerate(petersen_edges):
                vol = voltages[idx]
                for j in range(m):
                    a_v = u * m + j
                    b_v = v * m + ((j + vol) % m)
                    edges.append((a_v, b_v))
            
            if _check_girth_fast(n, edges, g):
                return n, edges
    
    return None

def _voltage_lift_dodecahedron(k, g, deadline, seed=123):
    """
    Voltage lift of the dodecahedron graph (20 vertices, girth 5).
    This might yield better girth/size ratios than Petersen (10 vertices, girth 5)
    because the base graph is larger, so the lift is larger but the "effective"
    girth gain per vertex might be better? Or worse?
    Actually, the girth of the lift is at least the girth of the base.
    If base has girth 5, lift has girth >= 5.
    To get girth 16, we need to avoid cycles of length < 16.
    The number of cycles in the base grows exponentially.
    Dodecahedron has 20 vertices, 30 edges.
    Petersen has 10 vertices, 15 edges.
    Lift size: 20*m vs 10*m.
    So dodecahedron lift is 2x larger.
    But maybe the voltage constraints are easier to satisfy?
    Let's try it.
    """
    if k != 3:
        return None
    
    random.seed(seed)
    
    # Dodecahedron graph construction
    # 20 vertices: 2 outer pentagons, 2 inner pentagons, 2 middle pentagons?
    # Standard construction:
    # Outer ring: 0-4
    # Inner ring: 5-9 (connected to outer)
    # Middle ring: 10-14
    # Inner ring: 15-19
    # Actually, let's use a known list of edges for dodecahedron.
    
    dodeca_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 6), (6, 7), (7, 8), (8, 9), (9, 5),
        (10, 11), (11, 12), (12, 13), (13, 14), (14, 10),
        (15, 16), (16, 17), (17, 18), (18, 19), (19, 15),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),
        (5, 10), (6, 11), (7, 12), (8, 13), (9, 14),
        (10, 15), (11, 16), (12, 17), (13, 18), (14, 19)
    ]
    
    n_base = 20
    m_edges = len(dodeca_edges)
    
    adj_base = [[] for _ in range(n_base)]
    for i, (u, v) in enumerate(dodeca_edges):
        adj_base[u].append((v, i))
        adj_base[v].append((u, i))
    
    cycles = []
    seen = set()
    
    def dfs_cycles(start, curr, visited, edge_path, length):
        if length > 15: # We only care about cycles up to length 15 for girth 16
            return
        for next_v, edge_idx in adj_base[curr]:
            if next_v == start and length >= 3:
                cycle_key = tuple(sorted(edge_path))
                if cycle_key not in seen:
                    seen.add(cycle_key)
                    cycles.append(cycle_key)
            elif next_v not in visited:
                visited.add(next_v)
                dfs_cycles(start, next_v, visited, edge_path + [edge_idx], length + 1)
                visited.remove(next_v)
    
    for start in range(n_base):
        visited = {start}
        dfs_cycles(start, start, visited, [], 0)
    
    primes_to_try = [5, 7, 11, 13, 17, 19, 23]
    
    for m in primes_to_try:
        if time.time() > deadline:
            break
        
        for a in range(1, m):
            for b in range(m):
                voltages = [(a * i + b) % m for i in range(m_edges)]
                
                valid = True
                for C in cycles:
                    s = sum(voltages[i] for i in C) % m
                    if s == 0:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                n = 20 * m
                edges = []
                for idx, (u, v) in enumerate(dodeca_edges):
                    vol = voltages[idx]
                    for j in range(m):
                        a_v = u * m + j
                        b_v = v * m + ((j + vol) % m)
                        edges.append((a_v, b_v))
                
                if _check_girth_fast(n, edges, g):
                    return n, edges
        
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            voltages = [random.randint(1, m - 1) for _ in range(m_edges)]
            
            valid = True
            for C in cycles:
                s = sum(voltages[i] for i in C) % m
                if s == 0:
                    valid = False
                    break
            
            if not valid:
                continue
            
            n = 20 * m
            edges = []
            for idx, (u, v) in enumerate(dodeca_edges):
                vol = voltages[idx]
                for j in range(m):
                    a_v = u * m + j
                    b_v = v * m + ((j + vol) % m)
                    edges.append((a_v, b_v))
            
            if _check_girth_fast(n, edges, g):
                return n, edges
    
    return None

def construct(k, g):
    start_time = time.time()
    deadline = 140.0

    if k != 3:
        return _luw_construct(k, g)

    # Try Cayley graph search first
    result = _cayley_search(k, g, deadline - 40)
    if result is not None:
        n, edges = result
        if _check_girth_fast(n, edges, g):
            return n, edges

    # Try voltage lift of Petersen
    result = _voltage_lift_petersen(k, g, deadline - 20, seed=42)
    if result is not None:
        n, edges = result
        if _check_girth_fast(n, edges, g):
            return n, edges

    # Try voltage lift of Dodecahedron (structural change)
    result = _voltage_lift_dodecahedron(k, g, deadline - 10, seed=123)
    if result is not None:
        n, edges = result
        if _check_girth_fast(n, edges, g):
            return n, edges

    # Fallback to LUW
    return _luw_construct(k, g)
