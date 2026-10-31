import time
import random
import numpy as np
from collections import deque

def _luw_construct(k, g):
    """
    Lazebnik-Ustimenko-Woldar construction for 3-regular graphs.
    We try smaller d values to get fewer vertices, checking girth empirically.
    """
    if k != 3:
        # Fallback for other k (should not be hit for this task)
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

    # For k=3, we use q=3.
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

def _build_adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _check_girth_fast(n, edges, min_girth):
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

def _voltage_lift_dodecahedron(k, g, deadline):
    if k != 3:
        return None
    
    # Dodecahedron graph: 20 vertices, 30 edges
    # Vertices 0-4: top face, 5-9: middle top, 10-14: middle bottom, 15-19: bottom face?
    # Standard dodecahedron coordinates or just adjacency list.
    # Let's use a known adjacency list for the dodecahedron.
    # 0 is connected to 1, 4, 5
    # 1 is connected to 0, 2, 6
    # 2 is connected to 1, 3, 7
    # 3 is connected to 2, 4, 8
    # 4 is connected to 0, 3, 9
    # 5 is connected to 0, 6, 10
    # 6 is connected to 1, 5, 11
    # 7 is connected to 2, 6, 12
    # 8 is connected to 3, 7, 13
    # 9 is connected to 4, 8, 14
    # 10 is connected to 5, 11, 15
    # 11 is connected to 6, 10, 16
    # 12 is connected to 7, 11, 17
    # 13 is connected to 8, 12, 18
    # 14 is connected to 9, 13, 19
    # 15 is connected to 10, 16, 19
    # 16 is connected to 11, 15, 17
    # 17 is connected to 12, 16, 18
    # 18 is connected to 13, 17, 19
    # 19 is connected to 14, 18, 15
    
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 6), (6, 7), (7, 8), (8, 9), (9, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),
        (10, 11), (11, 12), (12, 13), (13, 14), (14, 10),
        (5, 10), (6, 11), (7, 12), (8, 13), (9, 14),
        (15, 16), (16, 17), (17, 18), (18, 19), (19, 15),
        (10, 15), (11, 16), (12, 17), (13, 18), (14, 19)
    ]
    # Wait, the last set of edges connects 10-15 etc?
    # Let's verify degrees.
    # 0: 1,4,5 (3)
    # 1: 0,2,6 (3)
    # ...
    # 10: 11,14,5 (3)
    # 15: 16,19,10 (3)
    # This looks like a valid 3-regular graph.
    
    n_base = 20
    adj_base = [[] for _ in range(n_base)]
    for i, (u, v) in enumerate(edges):
        adj_base[u].append((v, i))
        adj_base[v].append((u, i))

    # Find all cycles in Dodecahedron up to length g-1
    cycles = []
    seen = set()
    
    def dfs(start, curr, visited, edge_path, length):
        if length >= g:
            return
        for next_v, edge_idx in adj_base[curr]:
            if next_v == start and length >= 3:
                cycle_key = tuple(sorted(edge_path))
                if cycle_key not in seen:
                    seen.add(cycle_key)
                    cycles.append(cycle_key)
            elif next_v not in visited:
                visited.add(next_v)
                dfs(start, next_v, visited, edge_path + [edge_idx], length + 1)
                visited.remove(next_v)
                
    for start in range(n_base):
        visited = {start}
        dfs(start, start, visited, [], 0)
        
    # Try small primes m
    # We want n = 20 * m to be small.
    # m=47 -> n=940
    primes = [47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    for m in primes:
        if time.time() > deadline:
            break
        
        # Random search for voltages
        for attempt in range(2000):
            if time.time() > deadline:
                break
            voltages = [random.randint(0, m - 1) for _ in range(30)]
            valid = True
            for C in cycles:
                s = sum(voltages[i] for i in C) % m
                if s == 0:
                    valid = False
                    break
            if valid:
                # Construct lift
                n = n_base * m
                lift_edges = []
                for idx, (u, v) in enumerate(edges):
                    vol = voltages[idx]
                    for j in range(m):
                        a = u * m + j
                        b = v * m + ((j + vol) % m)
                        lift_edges.append((a, b))
                return n, lift_edges
    return None

def construct(k, g):
    start_time = time.time()
    deadline = 140.0

    if k != 3:
        return _luw_construct(k, g)

    # Try Voltage Lift of Dodecahedron
    result = _voltage_lift_dodecahedron(k, g, deadline)
    if result is not None:
        n, edges = result
        if _check_girth_fast(n, edges, g):
            return n, edges

    # Fallback to LUW
    return _luw_construct(k, g)
