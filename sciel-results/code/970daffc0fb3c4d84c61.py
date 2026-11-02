import time
import random
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with girth at least g on as few vertices as possible.
    Returns (n, edges) where edges is a list of (u, v) on vertices 0..n-1.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    if k == 4 and g == 10:
        # 1. Try Cayley graph on Z_16 x Z_20
        result = _try_cayley_320(deadline)
        if result is not None:
            return result
        
        # 2. Try Randomized Local Search for 320 vertices
        result = _try_local_search_320(deadline)
        if result is not None:
            return result

    # Fallback: LUW Construction
    return _luw_construct(k, g)

def _try_cayley_320(deadline):
    m, n = 16, 20
    total = m * n
    
    def check_girth(x, y):
        gens = [
            (1, 0),
            (m-1, 0),
            (x, y),
            ((-x)%m, (-y)%n)
        ]
        
        visited = {}
        visited[(0,0)] = 0
        queue = deque([(0,0)])
        
        while queue:
            u, v = queue.popleft()
            depth = visited[(u,v)]
            
            for g1, g2 in gens:
                nu = (u + g1) % m
                nv = (v + g2) % n
                if (nu, nv) not in visited:
                    visited[(nu, nv)] = depth + 1
                    queue.append((nu, nv))
                else:
                    cycle_len = depth + 1 + visited[(nu, nv)]
                    if cycle_len < 10:
                        return False
        return True

    for x in range(m):
        if time.time() > deadline:
            return None
        for y in range(n):
            if time.time() > deadline:
                return None
            
            if x == 0 and y == 0: continue
            if x == 1 and y == 0: continue
            if x == m-1 and y == 0: continue
            if (2*x) % m == 0 and (2*y) % n == 0: continue
            
            if check_girth(x, y):
                edges = []
                for u in range(m):
                    for v in range(n):
                        nu = (u + 1) % m
                        nv = v
                        if (u, v) < (nu, nv):
                            edges.append(((u*m+v, nu*m+nv)))
                        
                        nu = (u + x) % m
                        nv = (v + y) % n
                        if (u, v) < (nu, nv):
                            edges.append(((u*m+v, nu*m+nv)))
                
                deg = [0]*total
                for e1, e2 in edges:
                    deg[e1] += 1
                    deg[e2] += 1
                if all(d == 4 for d in deg):
                    return total, edges

    return None

def _try_local_search_320(deadline):
    n = 320
    # Initialize random 4-regular graph
    # This is a simplified heuristic. 
    # Given the complexity, this might not find the solution in time.
    # But it's better than nothing if Cayley fails.
    
    # For now, return None to avoid wasting time if not implemented robustly.
    # The LUW fallback is guaranteed to be valid.
    return None

def _luw_construct(k, g):
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
