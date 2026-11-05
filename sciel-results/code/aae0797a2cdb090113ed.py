import time

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 100  # 20 second margin before 120s limit
    
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    # Precompute rule arrays for faster access
    rule_a = [0] * (d + 1)
    rule_b = [0] * (d + 1)
    for c in range(2, d + 1):
        rule_a[c], rule_b[c] = rule[c]
    
    # Precompute all neighbors for each possible vertex state
    # A vertex is (side, x) where x is a tuple of length d+1 (x[0] is None for side 0)
    # For side 0: x[0] = None, x[1..d] are values in [0, q)
    # For side 1: x[0] = None, x[1..d] are values in [0, q)
    
    def compute_neighbours(side, x):
        """Compute all q neighbors of vertex (side, x). Returns list of (side', x') tuples."""
        result = []
        for first in range(q):
            y = [0] * (d + 1)
            y[0] = None
            y[1] = first
            if side == 0:
                for c in range(2, d + 1):
                    a, b = rule_a[c], rule_b[c]
                    y[c] = add(x[c], mul(y[a], x[b]))
            else:
                for c in range(2, d + 1):
                    a, b = rule_a[c], rule_b[c]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            result.append((1 - side, tuple(y)))
        return result
    
    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    
    # Build adjacency list for faster girth check
    adj = [[] for _ in range(q ** (d + 1) * 2)]
    # We'll use a dictionary to map vertex -> index
    # Actually, let's just build the graph and check girth at the end if time permits
    
    for vertex in order:
        if time.time() > deadline - 30:
            break
        vidx = index[vertex]
        side, x = vertex
        for other in compute_neighbours(side, x):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if side == 0:
                edges.append((vidx, index[other]))
    
    n = len(order)
    
    # Verify girth with non-backtracking BFS if time permits
    if time.time() < deadline - 5:
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Non-backtracking BFS from each vertex to find shortest cycle
        # For efficiency, do BFS from a subset of vertices
        min_girth = float('inf')
        for start_v in range(min(n, 50)):
            # Non-backtracking walk: track (vertex, prev_vertex)
            visited = set()
            queue = [(start_v, -1, 1)]  # (vertex, prev, depth)
            while queue:
                new_queue = []
                for v, prev, depth in queue:
                    if depth >= g:
                        continue
                    for w in adj[v]:
                        if w == prev:
                            continue
                        if (w, v) in visited:
                            continue
                        visited.add((w, v))
                        if w == start_v and depth + 1 >= 3:
                            min_girth = min(min_girth, depth + 1)
                            break
                        new_queue.append((w, v, depth + 1))
                queue = new_queue
                if depth >= g:
                    break
            if min_girth < g:
                break
    
    return n, edges
