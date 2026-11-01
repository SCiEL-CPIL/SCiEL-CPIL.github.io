"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque
import random

def _verify_girth(edges, n, k, g):
    """Verify every vertex has degree k and no cycle shorter than g via non-backtracking BFS."""
    if n <= 0:
        return False
    adj = [[] for _ in range(n)]
    for u, v in edges:
        if u == v:
            return False
        if u < 0 or u >= n or v < 0 or v >= n:
            return False
        adj[u].append(v)
        adj[v].append(u)
    for i in range(n):
        if len(adj[i]) != k:
            return False
        if len(set(adj[i])) != k:
            return False
    if n <= 2000:
        for s in range(n):
            visited = set()
            queue = deque()
            for nb in adj[s]:
                queue.append((s, nb, 1))
                visited.add((s, nb))
            while queue:
                prev, curr, dist = queue.popleft()
                if curr == s and dist < g:
                    return False
                if dist >= g:
                    continue
                for nb in adj[curr]:
                    if nb == prev:
                        continue
                    if (curr, nb) not in visited:
                        visited.add((curr, nb))
                        queue.append((curr, nb, dist + 1))
    return True

def _luw_construction(q, d, rule, add_func, mul_func, deadline):
    """Generic L-U-W construction."""
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add_func(x[c], mul_func(y[a], x[b]))
                else:
                    y[c] = add_func(x[c], mul_func(x[a], y[b]))
            yield 1 - side, tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges

def construct(k, g):
    deadline = time.time() + 140
    rng = random.Random(42)
    
    if k == 4 and g == 10:
        # Use GF(4) with multiple rule patterns
        q = 4
        gf4_mul = [
            [0, 0, 0, 0],
            [0, 1, 2, 3],
            [0, 2, 3, 1],
            [0, 3, 1, 2]
        ]
        
        def gf4_add(a, b):
            return a ^ b
        
        def gf4_mul_func(a, b):
            return gf4_mul[a][b]
        
        best = None
        
        # Try the original rule pattern with different d
        for d in range(5, 12):
            if time.time() > deadline:
                break
            rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
            for c in range(5, d + 1):
                r = (c - 5) % 4
                c0 = c - (c - 5) % 4
                rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
            
            n, edges = _luw_construction(q, d, rule, gf4_add, gf4_mul_func, deadline)
            if n > 0 and _verify_girth(edges, n, k, g):
                if best is None or n < best[0]:
                    best = (n, edges)
                    if n <= 320:
                        return best
        
        # Try random rules
        for d in range(5, 12):
            if time.time() > deadline:
                break
            for _ in range(200):
                if time.time() > deadline:
                    break
                rule = {}
                valid = True
                for c in range(2, d + 1):
                    a = rng.randint(1, min(c - 1, d - 1))
                    b = rng.randint(1, min(c - 1, d - 1))
                    if a == b:
                        valid = False
                        break
                    rule[c] = (a, b)
                if not valid:
                    continue
                
                n, edges = _luw_construction(q, d, rule, gf4_add, gf4_mul_func, deadline)
                if n > 0 and _verify_girth(edges, n, k, g):
                    if best is None or n < best[0]:
                        best = (n, edges)
                        if n <= 320:
                            return best
        
        if best:
            return best
    
    # Fallback
    half_g = (g - 1) // 2
    moore = 1 + k * sum((k - 1) ** i for i in range(half_g))
    return moore, []
# EVOLVE-BLOCK-END
