import time
import random
from collections import deque
import numpy as np

def _build_adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _check_girth_fast(n, edges, min_girth):
    """Non-backtracking BFS girth check from every vertex."""
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

    # Try smaller d values first to get smaller graphs
    for d in [7, 8, 9, 10, 11, 12]:
        n, edges = build_luw(d)
        if _check_girth_fast(n, edges, g):
            return n, edges
    return build_luw(12)

def _find_cycles_dfs(base_edges, max_len, n_base):
    """Find all cycles up to max_len in a small base graph."""
    adj_base = [[] for _ in range(n_base)]
    for i, (u, v) in enumerate(base_edges):
        adj_base[u].append((v, i))
        adj_base[v].append((u, i))

    cycles = []
    seen = set()

    def dfs(start, curr, visited, edge_path, length):
        if length >= max_len:
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
    return cycles

def _voltage_lift_small_base(base_edges, n_base, g, deadline):
    """Voltage lift of a small base graph to get 3-regular graph with girth >= g."""
    n_edges = len(base_edges)
    cycles = _find_cycles_dfs(base_edges, g - 1, n_base)
    
    # Try primes m; n = n_base * m
    primes = [13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
    
    # Moore bound for (3,g): for g=16, it's 2*(2^7-1) = 254
    moore_bound = 2 * (2 ** ((g // 2) - 1) - 1) if g % 2 == 0 else 2 * (2 ** (g // 2) - 1)
    
    for m in primes:
        if time.time() > deadline:
            break
        n = n_base * m
        if n < moore_bound:
            continue
        if n > 5000:
            break
            
        for attempt in range(3000):
            if time.time() > deadline:
                break
            voltages = [random.randint(0, m - 1) for _ in range(n_edges)]
            valid = True
            for C in cycles:
                s = sum(voltages[i] for i in C) % m
                if s == 0:
                    valid = False
                    break
            if valid:
                lift_edges = []
                for idx, (u, v) in enumerate(base_edges):
                    vol = voltages[idx]
                    for j in range(m):
                        a = u * m + j
                        b = v * m + ((j + vol) % m)
                        lift_edges.append((a, b))
                return n, lift_edges
    return None

def _dodecahedron_edges():
    """Return edges of the dodecahedron graph (20 vertices, 30 edges)."""
    # Dodecahedron: 20 vertices, 30 edges, 3-regular
    # Using a standard embedding
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 6), (6, 7), (7, 8), (8, 9), (9, 5),
        (10, 11), (11, 12), (12, 13), (13, 14), (14, 10),
        (15, 16), (16, 17), (17, 18), (18, 19), (19, 15),
        (0, 15), (1, 10), (2, 11), (3, 12), (4, 13),
        (5, 14), (6, 5), (7, 16), (8, 17), (9, 18),
        (10, 19), (11, 0), (12, 1), (13, 2), (14, 3),
        (15, 4)
    ]
    # Remove duplicates and self-loops
    unique = set()
    result = []
    for u, v in edges:
        if u != v:
            key = (min(u, v), max(u, v))
            if key not in unique:
                unique.add(key)
                result.append(key)
    return result

def _cage_construction(k, g, deadline):
    """
    Try multiple construction strategies to find the smallest 3-regular graph
    with girth at least g.
    """
    if k != 3:
        return _luw_construct(k, g)
    
    results = []
    
    # Strategy 1: LUW construction with various d values
    n_luw, edges_luw = _luw_construct(k, g)
    if _check_girth_fast(n_luw, edges_luw, g):
        results.append((n_luw, edges_luw))
    
    # Strategy 2: Voltage lift of Petersen graph
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    result = _voltage_lift_small_base(petersen_edges, 10, g, deadline)
    if result is not None:
        n, edges = result
        if _check_girth_fast(n, edges, g):
            results.append((n, edges))
    
    # Strategy 3: Voltage lift of dodecahedron graph
    dodec_edges = _dodecahedron_edges()
    result = _voltage_lift_small_base(dodec_edges, 20, g, deadline)
    if result is not None:
        n, edges = result
        if _check_girth_fast(n, edges, g):
            results.append((n, edges))
    
    if results:
        return min(results, key=lambda x: x[0])
    
    # Fallback: return LUW
    return n_luw, edges_luw

def construct(k, g):
    start_time = time.time()
    deadline = 135.0  # Safety margin before 150s kill
    
    result = _cage_construction(k, g, deadline)
    return result
