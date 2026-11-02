import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = 145.0
    
    if k != 4 or g != 10:
        return _construct_lw(k, g)
    
    # Try to construct the theoretical minimum n=244 (m=61)
    # Using voltage graph on C4 with m=61 and voltages (1,1,1,1)
    m = 61
    v01, v12, v23, v30 = 1, 1, 1, 1
    
    n = 4 * m
    adj = [[] for _ in range(n)]
    
    # Build lift
    for a in range(m):
        u1 = 0 * m + a
        v1 = 1 * m + (a + v01) % m
        adj[u1].append(v1)
        adj[v1].append(u1)
        
        u2 = 1 * m + a
        v2 = 2 * m + (a + v12) % m
        adj[u2].append(v2)
        adj[v2].append(u2)
        
        u3 = 2 * m + a
        v3 = 3 * m + (a + v23) % m
        adj[u3].append(v3)
        adj[v3].append(u3)
        
        u4 = 3 * m + a
        v4 = 0 * m + (a + v30) % m
        adj[u4].append(v4)
        adj[v4].append(u4)
        
    # Verify simplicity and degree
    is_simple = True
    edge_set = set()
    for i in range(n):
        for j in adj[i]:
            if i == j:
                is_simple = False
                break
            key = (min(i, j), max(i, j))
            if key in edge_set:
                is_simple = False
                break
            edge_set.add(key)
        if not is_simple:
            break
    
    if not is_simple:
        return _construct_lw(k, g)
        
    valid = True
    for i in range(n):
        if len(adj[i]) != k:
            valid = False
            break
            
    if not valid:
        return _construct_lw(k, g)
        
    # Verify girth
    if _verify_girth_fast(n, adj, g):
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        return n, edges
        
    # Fallback if the specific construction fails
    return _construct_lw(k, g)

def _verify_girth_fast(n, adj, g):
    """Fast girth verification using non-backtracking walks."""
    for start_v in range(n):
        visited = set()
        queue = deque()
        
        for nb in adj[start_v]:
            visited.add((nb, start_v))
            queue.append((nb, start_v, 1))
        
        while queue:
            curr, prev, depth = queue.popleft()
            
            if curr == start_v and depth < g:
                return False
            
            if depth >= g - 1:
                continue
            
            for nb in adj[curr]:
                if nb == prev:
                    continue
                if (nb, curr) not in visited:
                    visited.add((nb, curr))
                    queue.append((nb, curr, depth + 1))
    
    return True

def _construct_lw(k, g):
    """Fallback LW graph construction."""
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
    
    n = len(order)
    return n, edges
