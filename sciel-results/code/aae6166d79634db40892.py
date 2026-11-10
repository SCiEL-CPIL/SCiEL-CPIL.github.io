import random
import time
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin (evaluator kills at 150s)
    
    best_n = float('inf')
    best_edges = None
    
    # Moore bound for (5,9) is approximately 428
    moore_bound = 2 + k * sum((k-1)**i for i in range((g-2)//2))
    
    # Strategy: Try voltage graph lifts with various base graphs
    # Try multiple base graph sizes and voltage group sizes
    
    candidates = []
    for v_b in range(k + 1, 40):
        if (k * v_b) % 2 != 0:
            continue
        for m in range(20, 400):
            n = v_b * m
            if n < moore_bound:
                continue
            if n > 1500:
                break
            candidates.append((v_b, m))
    
    random.shuffle(candidates)
    
    for v_b, m in candidates:
        if time.time() > deadline:
            break
            
        # Try multiple random base graphs
        for trial in range(3):
            if time.time() > deadline:
                break
            graph = _random_regular_graph(v_b, k)
            if graph is None:
                continue
            
            edges_b = graph
            n_edges = len(edges_b)
            
            # Precompute closed walks up to length g-1 in base graph
            closed_walks = _find_closed_walks(edges_b, v_b, g - 1)
            
            # Try random voltage assignments with constraint checking
            max_attempts = 30
            for attempt in range(max_attempts):
                if time.time() > deadline:
                    break
                    
                voltages = [random.randint(0, m - 1) for _ in range(n_edges)]
                
                # Quick check: verify no short closed walk has zero voltage sum
                valid = True
                for walk in closed_walks:
                    voltage_sum = 0
                    for idx in walk:
                        voltage_sum += voltages[idx]
                    if voltage_sum % m == 0:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                derived_edges = _construct_derived_graph_correct(edges_b, voltages, m)
                n = v_b * m
                
                if _check_girth_fast(derived_edges, n, k, g):
                    return n, derived_edges
    
    # Fallback: LUW construction
    return _luw_construct(k, g)

def _random_regular_graph(v, k):
    """Generate a random k-regular simple graph on v vertices."""
    if (v * k) % 2 != 0:
        return None
    if k >= v:
        return None
    if k == v - 1:
        # Complete graph
        return [(i, j) for i in range(v) for j in range(i+1, v)]
    
    # Configuration model with retries
    for _ in range(200):
        stubs = []
        for i in range(v):
            stubs.extend([i] * k)
        random.shuffle(stubs)
        
        edges = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u, v2 = stubs[i], stubs[i+1]
            if u == v2:
                valid = False
                break
            edge = (min(u, v2), max(u, v2))
            if edge in edges:
                valid = False
                break
            edges.add(edge)
        
        if valid:
            return list(edges)
    
    return None

def _find_closed_walks(edges, n, max_len):
    """Find all closed walks of length up to max_len in a graph.
    
    Returns list of edge index sequences for each closed walk.
    """
    # Build adjacency with edge indices
    adj = [[] for _ in range(n)]
    for idx, (u, v) in enumerate(edges):
        adj[u].append((v, idx))
        adj[v].append((u, idx))
    
    closed_walks = []
    
    # DFS from each vertex
    for start in range(n):
        # Walk: (current_vertex, path_of_edge_indices)
        stack = [(start, [])]
        while stack:
            vertex, path = stack.pop()
            if len(path) > 0 and vertex == start and len(path) >= 3:
                # Found a closed walk
                closed_walks.append(path[:])
                continue
            if len(path) >= max_len:
                continue
            for neighbor, edge_idx in adj[vertex]:
                if len(path) == 0:
                    # First step from start
                    if neighbor != start:
                        stack.append((neighbor, [edge_idx]))
                else:
                    # Avoid immediate backtracking
                    if edge_idx != path[-1]:
                        stack.append((neighbor, path + [edge_idx]))
    
    return closed_walks

def _construct_derived_graph_correct(base_edges, voltages, m):
    """Construct the derived graph from a voltage graph.
    
    For each undirected edge (u, v) in the base graph with voltage w:
    - (u, a) is connected to (v, (a + w) % m)
    
    Since the graph is undirected, we only add each edge once.
    """
    v_b = max([max(e) for e in base_edges]) + 1 if base_edges else 1
    derived_edges = []
    
    for idx, (u, v) in enumerate(base_edges):
        w = voltages[idx]
        for a in range(m):
            u_idx = u * m + a
            v_idx = v * m + (a + w) % m
            derived_edges.append((u_idx, v_idx))
    
    return derived_edges

def _check_girth_fast(edges, n, k, g):
    """Check if the graph has girth >= g using BFS from each vertex.
    
    For efficiency, we only need to check up to depth (g-1)//2.
    If we find a cycle of length < g, return False.
    """
    if not edges:
        return True
    
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check degree
    for i in range(n):
        if len(adj[i]) != k:
            return False
    
    max_depth = (g - 1) // 2
    
    for s in range(n):
        # BFS from s
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        queue = deque([s])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= max_depth:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif parent[v] != u and parent[u] != v:
                    # Found a cycle: u - v where both are visited
                    # Cycle length = dist[u] + dist[v] + 1
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
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
