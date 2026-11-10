"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START
def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k."""
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

def _check_girth(n, edges, min_girth):
    """Check if graph has no cycle shorter than min_girth using BFS from each vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    for start in range(n):
        # BFS tracking (node, parent, distance)
        visited = {start: (None, 0)}
        queue = deque([(start, None, 0)])
        while queue:
            node, parent, dist = queue.popleft()
            if dist >= min_girth // 2:
                continue
            for nb in adj[node]:
                if nb == parent:
                    continue
                if nb in visited:
                    # Found a cycle of length dist + 1 + visited[nb][1]
                    cycle_len = dist + 1 + visited[nb][1]
                    if cycle_len < min_girth:
                        return False
                else:
                    visited[nb] = (node, dist + 1)
                    queue.append((nb, node, dist + 1))
    return True

def construct(k, g):
    deadline = time.time() + 140  # 10s safety margin
    best_n, best_edges = _luw_construct(k, g)
    
    # Try to find a smaller graph via randomized construction
    # Start from the Moore bound and go up
    # Moore bound for (k, g) with g odd: k * (k-1)^((g-1)/2) / 2 + 1
    g_half = (g - 1) // 2
    moore_bound = k * (k - 1) ** g_half // 2 + 1
    
    # Try a few sizes
    for n in range(max(moore_bound, 100), min(best_n, 1200) + 1, 10):
        if time.time() > deadline:
            break
        # Try a few random constructions
        for attempt in range(20):
            if time.time() > deadline:
                break
            # Generate a random k-regular graph
            try:
                edges = _random_k_regular(n, k)
                if edges is None:
                    continue
                if _check_girth(n, edges, g):
                    return n, edges
            except Exception:
                continue
    
    return best_n, best_edges

def _random_k_regular(n, k):
    """Generate a random k-regular graph using configuration model."""
    if n * k % 2 != 0:
        return None
    if k >= n:
        return None
    
    # Configuration model
    half_edges = n * k // 2
    stubs = []
    for i in range(n):
        stubs.extend([i] * k)
    random.shuffle(stubs)
    
    # Pair up stubs
    edges = set()
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i + 1]
        if u == v:
            return None  # Self-loop, retry
        edge = (min(u, v), max(u, v))
        if edge in edges:
            return None  # Multi-edge, retry
        edges.add(edge)
    
    return list(edges)
# EVOLVE-BLOCK-END
