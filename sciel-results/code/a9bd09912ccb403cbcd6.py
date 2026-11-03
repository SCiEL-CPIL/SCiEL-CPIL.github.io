import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 110s to be safe
    
    # Calculate Moore Bound
    if g % 2 == 0:
        r = g // 2
        n_min = k * ((k - 1) ** (r - 1))
    else:
        r = (g - 1) // 2
        n_min = 1 + k * sum((k - 1) ** i for i in range(r))
        
    # Ensure n_min is at least k+1
    n_min = max(n_min, k + 1)
    
    best_n = float('inf')
    best_edges = None
    
    # Try to find a graph with n < n_min + some margin
    # We will try increasing n
    # Step size 10
    max_n = 5000 # Cap to avoid infinite loop, LUW is ~6000
    
    for n in range(n_min, min(n_min + 2000, max_n), 10):
        if time.time() > deadline:
            break
            
        # Try multiple restarts for this n
        for attempt in range(20):
            if time.time() > deadline:
                break
                
            success, edges = try_build_graph(k, g, n)
            if success:
                if n < best_n:
                    best_n = n
                    best_edges = edges
                # Found a valid graph, can we find a smaller one?
                # We are iterating n upwards, so the first success is the smallest for this step.
                # But we might have skipped smaller n if they failed.
                # We should return immediately if we want the smallest n?
                # The loop goes upwards. So the first success is the best for this n.
                # But we started from n_min. So the first success overall is the best.
                return n, edges

    # If no graph found, fallback to LUW
    return construct_luw(k, g)

def try_build_graph(k, g, n):
    degree = [0] * n
    adj = [set() for _ in range(n)]
    edges = []
    
    # Helper to check if adding (u, v) creates a short cycle
    # We will do this by BFS from u to depth g-2
    # If v is in the BFS tree, reject.
    
    while sum(degree) < n * k:
        # Find a vertex u with degree < k
        # To reduce backtracking, pick u with lowest degree
        u_candidates = [i for i in range(n) if degree[i] < k]
        if not u_candidates:
            return False, None
            
        u = min(u_candidates, key=lambda i: degree[i])
        
        # BFS from u to depth g-2
        depth = g - 2
        visited = {u: 0}
        queue = [(u, 0)]
        while queue:
            node, d = queue.pop(0)
            if d >= depth:
                continue
            for nb in adj[node]:
                if nb not in visited:
                    visited[nb] = d + 1
                    queue.append((nb, d + 1))
        
        # Find valid v
        v_candidates = []
        for v in range(n):
            if degree[v] < k and v != u and v not in visited and (u, v) not in adj[u]:
                v_candidates.append(v)
                
        if not v_candidates:
            return False, None
            
        # Pick v with lowest degree to balance
        v = min(v_candidates, key=lambda i: degree[i])
        
        # Add edge
        adj[u].add(v)
        adj[v].add(u)
        degree[u] += 1
        degree[v] += 1
        edges.append((u, v))
        
    return True, edges

def construct_luw(k, g):
    # Copy of seed logic
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
