import time
import random
import numpy as np

# Keep the original LUW code as a fallback
def _luw_construct(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k: an explicit q-regular bipartite graph
    # whose girth is at least d + 5 for odd d. Points and lines are vectors over GF(q) (q prime,
    # or q = 4); the connected component of the zero point is returned. Valid, far from optimal.
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
                # point p ~ line l  iff  l[c] - p[c] = l[a] * p[b] for c = 2..d
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

def _generate_random_3reg(n, seed=None):
    if seed is not None:
        random.seed(seed)
    # Configuration model
    stubs = []
    for i in range(n):
        for _ in range(3):
            stubs.append(i)
    
    edges = set()
    attempts = 0
    while attempts < 100:
        random.shuffle(stubs)
        edges = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                valid = False
                break
            if (u, v) in edges or (v, u) in edges:
                valid = False
                break
            edges.add((u, v))
        if valid:
            return list(edges)
        attempts += 1
    return None

def _check_girth(n, edges, min_girth):
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # BFS from each vertex to check for cycles < min_girth
    # We only need to check up to depth floor((min_girth-1)/2)
    max_depth = (min_girth - 1) // 2
    
    for start in range(n):
        # visited[v] = depth
        visited = [-1] * n
        visited[start] = 0
        queue = [(start, 0, -1)] # node, depth, parent
        
        while queue:
            node, depth, parent = queue.pop(0)
            if depth > max_depth:
                continue
            for neighbor in adj[node]:
                if neighbor == parent:
                    continue
                if visited[neighbor] == -1:
                    visited[neighbor] = depth + 1
                    queue.append((neighbor, depth + 1, node))
                else:
                    # Found a cycle
                    # Cycle length is depth + visited[neighbor] + 1
                    # But we must ensure it's a simple cycle and not just a back edge to an ancestor
                    # In BFS, if we find a visited neighbor, the cycle length is depth + visited[neighbor] + 1
                    # However, this might not be the shortest cycle through that edge.
                    # But if ANY cycle is < min_girth, we fail.
                    cycle_len = depth + visited[neighbor] + 1
                    if cycle_len < min_girth:
                        return False
    return True

def construct(k, g):
    start_time = time.time()
    deadline = 110.0 # 10s margin
    
    # Try random search for smaller n
    # Heuristic: n should be around 2^(g/2) or larger
    # For g=16, 2^8=256 is Moore bound.
    # Random graphs need n >> Moore bound to have high girth.
    # Let's try n in [10000, 50000, 100000]
    
    target_ns = [20000, 40000, 60000, 80000, 100000]
    
    for n in target_ns:
        if time.time() > deadline:
            break
        for attempt in range(5):
            if time.time() > deadline:
                break
            edges = _generate_random_3reg(n)
            if edges is None:
                continue
            if _check_girth(n, edges, g):
                return n, edges
        
    # Fallback to LUW
    return _luw_construct(k, g)
