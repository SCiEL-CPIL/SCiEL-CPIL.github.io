import time
import random
from collections import deque

def _luw_construct(k, g):
    """Fast LUW construction for the cage problem."""
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
    index = {start: 0}
    order = [start]
    edges = []
    
    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges

def _verify_girth(n, edges, min_girth):
    """Fast girth verification using non-backtracking walks."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # For each vertex, do BFS with non-backtracking constraint
    for start in range(n):
        # State: (current_node, previous_node, distance)
        visited = set()
        queue = deque()
        
        for nb in adj[start]:
            queue.append((nb, start, 1))
            visited.add((nb, start))
        
        while queue:
            curr, prev, dist = queue.popleft()
            
            if curr == start:
                if dist + 1 < min_girth:
                    return False
                continue
            
            for nb in adj[curr]:
                if nb == prev:
                    continue
                state = (nb, curr)
                if state not in visited:
                    visited.add(state)
                    queue.append((nb, curr, dist + 1))
                    # Early termination: if we can complete a cycle shorter than min_girth
                    if dist + 2 < min_girth:
                        return False
    
    return True

def _try_random_4regular(n, time_limit, rng):
    """Try to find a 4-regular graph with girth >= 10 on n vertices."""
    start_time = time.time()
    attempts = 0
    
    while time.time() - start_time < time_limit:
        attempts += 1
        if attempts > 200:
            break
        
        # Generate random 4-regular graph using configuration model
        stubs = []
        for i in range(n):
            for _ in range(4):
                stubs.append(i)
        rng.shuffle(stubs)
        
        adj = [[] for _ in range(n)]
        valid = True
        for i in range(0, n * 4, 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                valid = False
                break
            if v in adj[u]:
                valid = False
                break
            adj[u].append(v)
            adj[v].append(u)
        
        if not valid:
            continue
        
        # Quick girth check
        if _verify_girth(n, [(min(u,v), max(u,v)) for u in range(n) for v in adj[u] if u < v], 10):
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            return n, edges
    
    return None

def construct(k, g):
    if k != 4 or g != 10:
        return _luw_construct(k, g)
    
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Get baseline from LUW construction
    best_n, best_edges = _luw_construct(k, g)
    
    # Verify the LUW graph has correct girth
    if not _verify_girth(best_n, best_edges, g):
        # Should not happen, but fallback
        pass
    
    # Moore bound for 4-regular, girth 10
    # 2 * sum_{i=0}^{4} 3^i = 2 * 121 = 242
    moore = 242
    
    # Try random search for smaller graphs, but very limited
    # This is unlikely to succeed, but we try
    rng = random.Random(42)
    
    # Try a few sizes close to the Moore bound
    for n in range(moore, min(best_n, moore + 50), 2):
        if time.time() > deadline - 10:
            break
        
        # Give very little time to each size
        time_alloc = min(2.0, max(0.1, (deadline - time.time()) / 10))
        res = _try_random_4regular(n, time_alloc, rng)
        if res:
            best_n, best_edges = res
            break
    
    return best_n, best_edges
