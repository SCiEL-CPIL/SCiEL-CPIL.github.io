"""Program for constructing small k-regular graphs of given girth."""
import numpy as np
from collections import deque
import random
import time

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Uses a greedy random search for k=3 to find small graphs,
    falling back to LUW construction for other cases or if search fails.
    """
    if k == 2:
        n = max(g, 3)
        edges = [(i, (i + 1) % n) for i in range(n)]
        return n, edges
    
    if k == 1:
        return 2, [(0, 1)]
    
    if k != 3:
        return _luw_construct(k, g)
    
    # For k=3, try random search
    n_moore = 2 * (2**(g//2) - 1) if g % 2 == 0 else 2 * (2**((g-1)//2) - 1)
    
    # Start search from a size where girth is likely >= g
    n_start = 2**(g-1)
    
    # Cap n_start to not be too huge if LUW is smaller
    n_luw = _luw_estimate_size(k, g)
    n_start = min(n_start, n_luw)
    
    best_n = n_luw
    best_edges = None
    
    start_time = time.time()
    time_limit = 100 # seconds
    
    n_try = n_start
    while n_try >= n_moore and time.time() - start_time < time_limit:
        success = False
        attempts = 0
        max_attempts = 50
        
        while attempts < max_attempts and time.time() - start_time < time_limit:
            edges = _generate_random_3_reg(n_try)
            if edges is None:
                break
            if _check_girth(n_try, edges, g):
                best_n = n_try
                best_edges = edges
                success = True
                break
            attempts += 1
            
        if success:
            n_try = n_try // 2
        else:
            break
            
    if best_edges is not None:
        return best_n, best_edges
    else:
        return _luw_construct(k, g)

def _luw_construct(k, g):
    """Original LUW construction logic."""
    if k == 2:
        n = max(g, 3)
        edges = [(i, (i + 1) % n) for i in range(n)]
        return n, edges
    if k == 1:
        return 2, [(0, 1)]
        
    candidates = []
    d_odd = g - 5
    if d_odd < 3: d_odd = 3
    if d_odd % 2 == 0: d_odd += 1
    candidates.append(d_odd)
    
    d_even = g - 4
    if d_even < 2: d_even = 2
    if d_even % 2 == 1: d_even += 1
    candidates.append(d_even)
    
    d = min(candidates)
    q = k
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        def add(x, y): return x ^ y
        def sub(x, y): return x ^ y
        def mul(x, y): return table[x][y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return (x * y) % q
    
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - r
        if r == 0:
            rule[c] = (1, c0 - 2)
        elif r == 1:
            rule[c] = (c0 - 1, 1)
        elif r == 2:
            rule[c] = (c0, 1)
        else:
            rule[c] = (1, c0 + 1)
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield (1 - side, tuple(y))
    
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
    
    n = len(order)
    return n, edges

def _luw_estimate_size(k, g):
    d = g - 5
    if d < 3: d = 3
    if d % 2 == 0: d += 1
    return 2 * (k ** (d - 1))

def _generate_random_3_reg(n):
    if n % 2 != 0 or n < 4:
        return None
    stubs = []
    for i in range(n):
        for _ in range(3):
            stubs.append(i)
    random.shuffle(stubs)
    edges = []
    seen = set()
    for i in range(0, n*3, 2):
        u, v = stubs[i], stubs[i+1]
        if u == v:
            return None
        if (min(u,v), max(u,v)) in seen:
            return None
        seen.add((min(u,v), max(u,v)))
        edges.append((u, v))
    return edges

def _check_girth(n, edges, g):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    for start in range(n):
        visited = [-1] * n
        parent = [-1] * n
        depth = [0] * n
        queue = deque([(start, 0)])
        visited[start] = 1
        
        while queue:
            u, d = queue.popleft()
            if d >= g // 2 - 1:
                continue
            for v in adj[u]:
                if visited[v] == 0:
                    visited[v] = 1
                    parent[v] = u
                    depth[v] = d + 1
                    queue.append((v, d + 1))
                else:
                    if v != parent[u]:
                        if depth[u] + depth[v] + 1 < g:
                            return False
    return True
# EVOLVE-BLOCK-END
