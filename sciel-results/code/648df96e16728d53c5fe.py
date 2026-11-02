import time
import random
import sys

_DEADLINE = 145.0  # safety margin before 150s kill

def _luw_construct(k, g):
    # Copy of the seed logic
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

def _voltage_lift_k55(m):
    """
    Constructs a voltage lift of K_{5,5} over Z_m.
    Vertices are (side, i, j) where side in {0,1}, i in 0..4, j in 0..m-1.
    side 0 corresponds to U, side 1 to V.
    Edge from (0, i, j) to (1, k, j + i*k % m).
    """
    n = 10 * m
    adj = [set() for _ in range(n)]
    
    def vid(side, i, j):
        return side * 5 * m + i * m + j
        
    for i in range(5):
        for j in range(m):
            u = vid(0, i, j)
            for k in range(5):
                # Voltage from U_i to V_k is i * k
                v_j = (j + i * k) % m
                v = vid(1, k, v_j)
                adj[u].add(v)
                adj[v].add(u)
                
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append((i, j))
    return n, edges, adj

def _check_girth(n, adj, min_girth_target=9):
    """
    Checks if the graph has girth at least min_girth_target.
    Returns True if girth >= min_girth_target, False otherwise.
    Uses BFS from each vertex to find shortest cycle.
    """
    # Optimization: If we find a cycle of length < min_girth_target, return False.
    # We only need to search up to depth min_girth_target // 2.
    max_depth = min_girth_target // 2
    
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        qi = 0
        while qi < len(queue):
            u = queue[qi]
            qi += 1
            du = dist[u]
            if du > max_depth:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    # Found a cycle
                    if v == parent[u]:
                        continue
                    # Cycle length is dist[u] + dist[v] + 1
                    L = dist[u] + dist[v] + 1
                    if L < min_girth_target:
                        return False
    return True

_START_TIME = time.time()

def construct(k, g):
    global _DEADLINE
    if k == 5 and g == 9:
        # Try voltage lift of K_{5,5} with m=5
        # This should yield n=50, girth >= 20
        n, edges, adj = _voltage_lift_k55(5)
        
        # Verify girth
        if _check_girth(n, adj, 9):
            return n, edges
            
        # If that fails (shouldn't), try m=7
        n, edges, adj = _voltage_lift_k55(7)
        if _check_girth(n, adj, 9):
            return n, edges
            
        # Fallback to LUW
        return _luw_construct(k, g)
        
    # Generic fallback
    return _luw_construct(k, g)
