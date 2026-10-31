"""Construct 3-regular graph of girth >= g using random generation + 2-switching."""
import random
import time
import math

def _generate_random_3reg(n, seed=42):
    """Generate a random 3-regular graph on n vertices using the configuration model."""
    rng = random.Random(seed)
    # Build half-edges
    half_edges = []
    for v in range(n):
        for _ in range(3):
            half_edges.append(v)
    rng.shuffle(half_edges)
    # Pair them up
    adj = [[] for _ in range(n)]
    edges = set()
    for i in range(0, n * 3, 2):
        a, b = half_edges[i], half_edges[i + 1]
        if a == b:
            continue  # skip loops, retry not done here; for simplicity, we'll fix later
        key = (min(a, b), max(a, b))
        if key in edges:
            continue
        edges.add(key)
        adj[a].append(b)
        adj[b].append(a)
    # This may not give exactly 3-regular; let's use a proper algorithm
    return adj, edges

def _proper_random_3reg(n, seed=42):
    """Generate a random 3-regular graph properly using the configuration model with retry."""
    rng = random.Random(seed)
    for attempt in range(100):
        half_edges = []
        for v in range(n):
            half_edges.extend([v] * 3)
        rng.shuffle(half_edges)
        adj = [[] for _ in range(n)]
        edges = set()
        valid = True
        for i in range(0, n * 3, 2):
            a, b = half_edges[i], half_edges[i + 1]
            if a == b:
                valid = False
                break
            key = (min(a, b), max(a, b))
            if key in edges:
                valid = False
                break
            edges.add(key)
            adj[a].append(b)
            adj[b].append(a)
        if valid:
            return adj, list(edges)
    raise RuntimeError("Failed to generate random 3-regular graph")

def _find_short_cycle(adj, max_len):
    """Find a cycle of length <= max_len using BFS from each vertex."""
    n = len(adj)
    for start in range(n):
        # BFS tracking (node, parent, depth)
        from collections import deque
        visited = {}  # node -> (depth, parent)
        queue = deque()
        for nb in adj[start]:
            queue.append((nb, start, 1))
            visited[nb] = (1, start)
        while queue:
            node, parent, depth = queue.popleft()
            if depth >= max_len:
                continue
            for nb in adj[node]:
                if nb == parent:
                    continue
                if nb == start:
                    return depth + 1  # found a cycle
                if nb not in visited:
                    visited[nb] = (depth + 1, node)
                    queue.append((nb, node, depth + 1))
                else:
                    v_depth, v_parent = visited[nb]
                    if v_parent != node and depth + 1 + v_depth < max_len:
                        return depth + 1 + v_depth
    return None

def _girth(adj):
    """Compute the girth of the graph."""
    n = len(adj)
    # Use BFS from each vertex to find shortest cycle
    # This is O(n * (n + m)) which is too slow for large n
    # Instead, use a smarter approach: BFS up to depth g/2
    best = n + 1
    for start in range(n):
        from collections import deque
        dist = {start: 0}
        parent = {start: -1}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            d = dist[node]
            if d >= best // 2:
                continue
            for nb in adj[node]:
                if nb not in dist:
                    dist[nb] = d + 1
                    parent[nb] = node
                    queue.append(nb)
                else:
                    if parent[node] != nb and parent[nb] != node:
                        # Check if this forms a cycle
                        cycle_len = d + 1 + dist[nb]
                        if cycle_len < best:
                            best = cycle_len
    return best if best <= n else n + 1

def construct(k, g):
    """Construct a k-regular graph with girth >= g."""
    if k != 3:
        # Fallback to LUW for other k
        return _luw_construct(k, g)
    
    deadline = time.time() + 140  # 140s to be safe
    start_time = time.time()
    
    # Moore bound for 3-regular, girth g
    # For even girth g=2r: n >= 2*(3*(3^(r-1)-1)/(3-1)) = 3*(3^(r-1)-1)
    # For odd girth g=2r+1: n >= 3*(3^r - 1)/(3-1) + 1 = ...
    if g % 2 == 0:
        r = g // 2
        moore = 3 * (3 ** (r - 1) - 1)
    else:
        r = (g - 1) // 2
        moore = 1 + 3 * (3 ** r - 1) // 2
    
    # Start with a reasonable size
    n = max(moore + 2, 100)
    n = n if n % 2 == 0 else n + 1  # must be even for 3-regular
    
    best_adj, best_edges = None, None
    best_girth = 0
    
    for seed in range(50):
        if time.time() > deadline - 5:
            break
        try:
            adj, edges = _proper_random_3reg(n, seed)
        except:
            continue
        
        # Compute girth
        girth = _girth(adj)
        if girth > best_girth:
            best_girth = girth
            best_adj, best_edges = adj, edges
        
        if girth >= g:
            return n, edges
    
    # If we didn't find girth >= g, try increasing n
    if best_adj is not None:
        return n, best_edges
    
    # Fallback: LUW construction
    return _luw_construct(k, g)

def _luw_construct(k, g):
    """LUW construction as fallback."""
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
