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


def _random_3regular(n, deadline, g):
    """
    Generate a random 3-regular graph on n vertices and check girth.
    Uses the configuration model with rejection.
    """
    if n % 2 != 0:
        return None
    
    best_n = None
    best_edges = None
    
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Configuration model: 3n half-edges
        half_edges = list(range(3 * n))
        random.shuffle(half_edges)
        
        # Pair up half-edges
        edges = []
        valid = True
        seen = set()
        for i in range(0, 3 * n, 2):
            u = half_edges[i] // 3
            v = half_edges[i + 1] // 3
            if u == v:
                valid = False
                break
            key = (min(u, v), max(u, v))
            if key in seen:
                valid = False
                break
            seen.add(key)
            edges.append((u, v))
        
        if not valid:
            continue
        
        # Check degree
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if any(d != 3 for d in deg):
            continue
        
        # Check girth
        if _check_girth_fast(n, edges, g):
            return n, edges
        
        if best_n is None or n < best_n:
            best_n = n
            best_edges = edges
    
    return None


def _construct_via_bipartite_lift(k, g, deadline):
    """
    Construct a 3-regular graph by lifting a bipartite base graph.
    Use a random 3-regular bipartite graph as base, then lift with voltages.
    """
    if k != 3:
        return None
    
    # Base: random 3-regular bipartite graph on 2*m vertices (m in each part)
    # Then lift by factor r to get 2*m*r vertices
    # Total n = 2*m*r
    
    # Try small m and r
    for m in [4, 5, 6, 7, 8, 10, 12, 15, 20]:
        if time.time() > deadline:
            break
        
        # Generate random 3-regular bipartite graph on 2*m vertices
        # Part A: 0..m-1, Part B: m..2m-1
        base_edges = []
        valid = False
        for _ in range(100):
            # Configuration model for bipartite
            left_half = list(range(3 * m))
            right_half = list(range(3 * m))
            random.shuffle(left_half)
            random.shuffle(right_half)
            
            base_edges = []
            seen = set()
            ok = True
            for i in range(3 * m):
                u = left_half[i] // 3
                v = m + right_half[i] // 3
                key = (u, v)
                if key in seen:
                    ok = False
                    break
                seen.add(key)
                base_edges.append((u, v))
            
            if ok:
                valid = True
                break
        
        if not valid:
            continue
        
        n_base = 2 * m
        e_base = len(base_edges)
        
        # Now lift by factor r
        for r in [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30]:
            if time.time() > deadline:
                break
            
            n = n_base * r
            # n must be even for 3-regular
            if n % 2 != 0:
                continue
            
            # Random voltages in Z_r
            for _ in range(50):
                if time.time() > deadline:
                    break
                voltages = [random.randint(0, r - 1) for _ in range(e_base)]
                
                edges = []
                for idx, (u, v) in enumerate(base_edges):
                    vol = voltages[idx]
                    for j in range(r):
                        a = u * r + j
                        b = v * r + ((j + vol) % r)
                        edges.append((a, b))
                
                if _check_girth_fast(n, edges, g):
                    return n, edges
    
    return None


def construct(k, g):
    start_time = time.time()
    deadline = 140.0  # 10s margin before 150s kill

    if k != 3:
        return _luw_construct(k, g)

    # Try bipartite lift approach (structural change)
    result = _construct_via_bipartite_lift(k, g, deadline)
    if result is not None:
        return result

    # Fallback to LUW
    return _luw_construct(k, g)
