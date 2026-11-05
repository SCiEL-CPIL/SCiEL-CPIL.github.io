import time
import random
import sys

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # 1. Try to find a small optimal graph via random search if Moore bound is small
    # This is effective for k <= 4 and small g
    moore_bound = 1 + k * sum((k - 1) ** i for i in range((g - 1) // 2))
    if moore_bound < 100:
        # Try random search for a few seconds
        best_n = None
        best_edges = None
        search_deadline = start_time + 5.0
        # Start from Moore bound, try up to moore_bound + 20
        max_n = min(moore_bound + 50, 100)
        for n in range(moore_bound, max_n + 1):
            if n * k % 2 != 0:
                continue
            if time.time() > search_deadline:
                break
            # Try to find a k-regular graph of girth g on n vertices
            for _ in range(1000): # 1000 attempts per n
                if time.time() > search_deadline:
                    break
                edges = generate_random_k_regular(n, k)
                if edges is None:
                    continue
                if has_girth_at_least(edges, n, g):
                    best_n = n
                    best_edges = edges
                    break
            if best_n is not None:
                break
        if best_n is not None:
            return best_n, best_edges

    # 2. Hardcoded Known Cages for specific small cases
    # Petersen Graph: k=3, g=5, n=10
    if k == 3 and g == 5:
        n = 10
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
        ]
        return n, edges
        
    # Heawood Graph: k=3, g=6, n=14
    if k == 3 and g == 6:
        n = 14
        # Incidence graph of Fano Plane
        # Points 0-6, Lines 7-13
        # Fano plane lines:
        # 012, 034, 056, 135, 146, 236, 245
        lines = [
            (0, 1, 2),
            (0, 3, 4),
            (0, 5, 6),
            (1, 3, 5),
            (1, 4, 6),
            (2, 3, 6),
            (2, 4, 5)
        ]
        edges = []
        for i, line in enumerate(lines):
            line_id = 7 + i
            for p in line:
                edges.append((p, line_id))
        return n, edges

    # 3. Lazebnik-Ustimenko-Woldar Graph (Fallback)
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

def generate_random_k_regular(n, k):
    if n * k % 2 != 0:
        return None
    # Configuration model
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
    random.shuffle(stubs)
    edges = set()
    for i in range(0, n * k, 2):
        u, v = stubs[i], stubs[i + 1]
        if u == v:
            continue # Skip loops for now, will result in non-regular or need repair
        if (u, v) in edges or (v, u) in edges:
            continue # Skip multi-edges
        edges.add((u, v))
    
    # Check if regular
    degree = [0] * n
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
    if all(d == k for d in degree):
        return list(edges)
    return None

def has_girth_at_least(edges, n, g):
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # BFS from each vertex to find shortest cycle
    # Optimization: Non-backtracking
    for start in range(n):
        # visited[node][prev] = distance
        # We want to find a cycle of length < g
        # Start with neighbors
        for neighbor in adj[start]:
            # Walk from neighbor, avoiding start initially
            # State: (current, prev, dist)
            # If we hit start with dist < g-1, we have a cycle of length dist+1
            # Actually, simpler: BFS from start, track parent
            # But standard BFS finds shortest path, not necessarily cycle through start.
            # To find girth, we can do BFS from each node, and if we find a back edge to a node at depth d,
            # cycle length is d + 1.
            # We just need to ensure no cycle < g.
            pass
    
    # Simpler check: Count non-backtracking walks?
    # Or just BFS from each node and check for cycles.
    # For small n (< 100), this is fast.
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        queue = [s]
        for u in queue:
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    # Found a cycle
                    # Length is dist[u] + dist[v] + 1
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

# EVOLVE-BLOCK-END
