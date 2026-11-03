import time
import random
import numpy as np

def construct(k, g):
    """
    Constructs a k-regular graph of girth at least g.
    Uses LUW construction as baseline, and attempts to find smaller graphs via random search.
    """
    start_time = time.time()
    deadline = start_time + 140.0

    # --- Part 1: LUW Construction (Baseline) ---
    # This is copied from the seed program
    q = k
    def girth_of_d(d):
        if d % 2 == 1:
            return d + 5
        else:
            return d + 6

    d = 3
    while girth_of_d(d) < g:
        d += 1

    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return x * y % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rules_cycle = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = rules_cycle[r]

    index = {}
    order = []
    edges = []

    start = (0, (None,) + (0,) * d)
    index[start] = 0
    order.append(start)

    for i, vertex in enumerate(order):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            neighbor = (1 - side, tuple(y))
            if neighbor not in index:
                index[neighbor] = len(order)
                order.append(neighbor)
            if side == 0:
                edges.append((index[vertex], index[neighbor]))
        
        if time.time() > deadline - 60.0: # Reserve time for search
             # If LUW generation is taking too long, we might skip search
             # But LUW for d=10 is fast.
             pass

    n_luw = len(order)
    best_n = n_luw
    best_edges = edges

    # --- Part 2: Random Search for Smaller Graphs ---
    # Only try if k=3 and g=16 (or similar small cases)
    # And if we have time.
    if k == 3 and g == 16:
        # Moore bound approx 510
        min_n = 512
        max_n = min(2000, n_luw) # Don't go above LUW
        
        # Helper to check girth
        def check_girth(edges_list, n_verts, target_girth):
            adj = [[] for _ in range(n_verts)]
            for u, v in edges_list:
                adj[u].append(v)
                adj[v].append(u)
            
            # BFS from each node
            for start in range(n_verts):
                visited = [-1] * n_verts
                parent = [-1] * n_verts
                visited[start] = 0
                queue = [start]
                while queue:
                    u = queue.pop(0)
                    if visited[u] >= target_girth // 2:
                        break
                    for v in adj[u]:
                        if visited[v] == -1:
                            visited[v] = visited[u] + 1
                            parent[v] = u
                            queue.append(v)
                        elif v != parent[u] and u != parent[v]:
                            # Found cycle
                            # Length is visited[u] + visited[v] + 1
                            cycle_len = visited[u] + visited[v] + 1
                            if cycle_len < target_girth:
                                return False
            return True

        def generate_random_3_reg(n):
            # Configuration model
            while True:
                stubs = list(range(3 * n))
                random.shuffle(stubs)
                edges = []
                valid = True
                for i in range(0, 3 * n, 2):
                    u = stubs[i] // 3
                    v = stubs[i+1] // 3
                    if u == v:
                        valid = False
                        break
                    if (u, v) in edges or (v, u) in edges:
                        valid = False
                        break
                    edges.append((u, v))
                if valid:
                    return edges
                # Retry
                if time.time() > deadline - 5.0:
                    return None

        # Try sizes
        for n in range(min_n, max_n, 2):
            if time.time() > deadline - 10.0:
                break
            
            # Try a few random graphs
            found = False
            for _ in range(50): # Max 50 attempts per n
                if time.time() > deadline - 5.0:
                    break
                edges_try = generate_random_3_reg(n)
                if edges_try is None:
                    continue
                if check_girth(edges_try, n, g):
                    best_n = n
                    best_edges = edges_try
                    found = True
                    break
            
            if found:
                break # Found smallest n, can return

    return best_n, best_edges
