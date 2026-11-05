import time
import random
import numpy as np

def _luw_construct(k, g):
    # Original LUW code from prompt
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

def _check_girth(n, edges, target_girth):
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    best_girth = float('inf')
    
    for start in range(n):
        # BFS
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        
        while head < len(queue):
            u = queue[head]
            head += 1
            
            # Pruning: if current depth is already >= best_girth, no need to go deeper
            if dist[u] >= best_girth:
                continue
                
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < best_girth:
                        best_girth = cycle_len
                        if best_girth < target_girth:
                            return best_girth
    return best_girth

def _generate_random_k_regular(n, k):
    # Returns list of edges or None if failed
    if (n * k) % 2 != 0:
        return None
        
    # Create stubs
    stubs = []
    for i in range(n):
        stubs.extend([i] * k)
        
    # Shuffle
    random.shuffle(stubs)
    
    # Pair
    edges = []
    edge_set = set()
    for i in range(0, len(stubs), 2):
        u = stubs[i]
        v = stubs[i+1]
        if u == v:
            return None # Loop
        if u > v:
            u, v = v, u
        if (u, v) in edge_set:
            return None # Multi-edge
        edge_set.add((u, v))
        edges.append((u, v))
        
    return edges

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Specific optimization for k=5, g=9
    if k == 5 and g == 9:
        # Start near the known best
        n = 1100
        step = 20
        max_attempts_per_n = 20
        
        best_n = None
        best_edges = None
        
        while time.time() < deadline:
            edges = None
            for _ in range(max_attempts_per_n):
                if time.time() > deadline:
                    break
                edges = _generate_random_k_regular(n, k)
                if edges is None:
                    continue
                
                # Check girth
                girth = _check_girth(n, edges, g)
                if girth >= g:
                    best_n = n
                    best_edges = edges
                    break
            
            if best_n is not None:
                return best_n, best_edges
            
            n += step
        
        if best_n is not None:
            return best_n, best_edges
        
        # Fallback to LUW if random search failed
        return _luw_construct(k, g)
    else:
        return _luw_construct(k, g)
