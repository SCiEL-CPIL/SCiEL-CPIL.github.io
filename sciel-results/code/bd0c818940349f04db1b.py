import numpy as np
import random
import time

def construct(k, g):
    deadline = time.time() + 140  # 10s safety margin before 150s kill
    
    if k != 4 or g != 10:
        # Fallback to LUW construction for other parameters
        return _luw_construct(k, g)
    
    # Build PG(2,3) incidence graph
    points, lines, line_points = _build_pg23()
    base_edges = []
    for i, p in enumerate(points):
        for j, lp in enumerate(line_points):
            if p in lp:
                base_edges.append((i, 13 + j))
    
    adj = {i: [] for i in range(26)}
    for u, v in base_edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Enumerate all 6-cycles and 8-cycles in the base graph
    cycles_6 = _enumerate_cycles(adj, 6)
    cycles_8 = _enumerate_cycles(adj, 8)
    
    m = 13  # Lift over Z_13: 26*13 = 338 vertices
    
    # Search for voltage assignment
    best_result = None
    for attempt in range(2000):
        if time.time() > deadline:
            break
        
        # Random voltage assignment
        voltages = {}
        for u, v in base_edges:
            voltages[(u, v)] = random.randint(0, m - 1)
            voltages[(v, u)] = (-voltages[(u, v)]) % m
        
        # Check if all 6-cycles and 8-cycles have non-zero voltage sum
        valid = True
        for cycle in cycles_6:
            s = 0
            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                s += voltages[(u, v)]
            if s % m == 0:
                valid = False
                break
        
        if valid:
            for cycle in cycles_8:
                s = 0
                for i in range(len(cycle)):
                    u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                    s += voltages[(u, v)]
                if s % m == 0:
                    valid = False
                    break
        
        if valid:
            # Build the lifted graph
            n_vertices = 26 * m
            lift_edges = []
            for u, v in base_edges:
                vu = voltages[(u, v)]
                for i in range(m):
                    lift_edges.append((u * m + i, v * m + (i + vu) % m))
            
            # Verify girth via BFS from each vertex
            if _verify_girth(n_vertices, lift_edges, g):
                return n_vertices, lift_edges
        
        # If no valid assignment found, try a smarter approach:
        # Use the structure to guide voltage assignment
        if attempt % 500 == 499 and time.time() < deadline - 30:
            # Try systematic search on a subset
            voltages = _systematic_voltages(base_edges, m, cycles_6, cycles_8)
            if voltages is not None:
                n_vertices = 26 * m
                lift_edges = []
                for u, v in base_edges:
                    vu = voltages[(u, v)]
                    for i in range(m):
                        lift_edges.append((u * m + i, v * m + (i + vu) % m))
                if _verify_girth(n_vertices, lift_edges, g):
                    return n_vertices, lift_edges
    
    # Fallback: LUW construction
    return _luw_construct(k, g)


def _build_pg23():
    """Build PG(2,3) points and lines."""
    points = []
    for x in range(3):
        for y in range(3):
            for z in range(3):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = 1 if x == 1 else 2
                    p = (1, (y * inv_x) % 3, (z * inv_x) % 3)
                elif y != 0:
                    inv_y = 1 if y == 1 else 2
                    p = (0, 1, (z * inv_y) % 3)
                else:
                    p = (0, 0, 1)
                if p not in points:
                    points.append(p)
    
    lines = []
    line_points = []
    for a in range(3):
        for b in range(3):
            for c in range(3):
                if a == 0 and b == 0 and c == 0:
                    continue
                if a != 0:
                    inv_a = 1 if a == 1 else 2
                    n = (1, (b * inv_a) % 3, (c * inv_a) % 3)
                elif b != 0:
                    inv_b = 1 if b == 1 else 2
                    n = (0, 1, (c * inv_b) % 3)
                else:
                    n = (0, 0, 1)
                
                if n in lines:
                    continue
                lines.append(n)
                pts = []
                for p in points:
                    if (n[0] * p[0] + n[1] * p[1] + n[2] * p[2]) % 3 == 0:
                        pts.append(p)
                line_points.append(pts)
    
    return points, lines, line_points


def _enumerate_cycles(adj, length):
    """Enumerate all cycles of given length in the base graph."""
    cycles = []
    n = len(adj)
    for start in range(n):
        # BFS/DFS to find cycles of exact length
        # Fix start vertex and first neighbor to avoid duplicates
        for first in adj[start]:
            if first < start:
                continue  # Avoid duplicate cycles
            # DFS to find paths of length-1 edges back to start
            stack = [(first, [start, first], {start, first})]
            while stack:
                node, path, visited = stack.pop()
                if len(path) == length:
                    if start in adj[node]:
                        cycles.append(tuple(path))
                    continue
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        stack.append((neighbor, path + [neighbor], visited | {neighbor}))
    return cycles


def _systematic_voltages(base_edges, m, cycles_6, cycles_8):
    """Try a more systematic voltage assignment."""
    # This is a placeholder for a smarter approach
    return None


def _verify_girth(n_vertices, edges, min_girth):
    """Verify that the graph has girth at least min_girth using BFS."""
    adj = [[] for _ in range(n_vertices)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check degree
    for i in range(n_vertices):
        if len(adj[i]) != 4:
            return False
    
    # BFS from each vertex to find shortest cycle
    for start in range(n_vertices):
        dist = [-1] * n_vertices
        parent = [-1] * n_vertices
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif parent[u] != v and parent[v] != u:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        return False
    return True


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction as fallback."""
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
