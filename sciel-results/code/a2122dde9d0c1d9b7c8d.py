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
        if u == v or u < 0 or u >= n or v < 0 or v >= n:
            return False
        adj[u].append(v)
        adj[v].append(u)
    for i in range(n):
        if len(adj[i]) != k or len(set(adj[i])) != k:
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
                    if nb != prev and (curr, nb) not in visited:
                        visited.add((curr, nb))
                        queue.append((curr, nb, dist + 1))
    return True

def _petersen_graph():
    n = 10
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 6), (6, 7), (7, 8), (8, 9), (9, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    return n, edges

def _lift_petersen(voltages, m):
    n_base, base_edges = _petersen_graph()
    n_lift = n_base * m
    edges = []
    for u, v in base_edges:
        if u < v:
            vol = voltages[(u, v)]
        else:
            vol = (-voltages[(v, u)]) % m
        for a in range(m):
            u_lift = u * m + a
            v_lift = v * m + ((a + vol) % m)
            edges.append((u_lift, v_lift))
    return n_lift, edges

def _check_girth_fast(edges, n, k, g):
    if n <= 0:
        return False
    adj = [[] for _ in range(n)]
    for u, v in edges:
        if u == v or u < 0 or u >= n or v < 0 or v >= n:
            return False
        adj[u].append(v)
        adj[v].append(u)
    for i in range(n):
        if len(adj[i]) != k:
            return False
    for s in range(n):
        visited = {}
        queue = deque()
        for nb in adj[s]:
            queue.append((s, nb, 1))
            visited[(s, nb)] = 1
        while queue:
            prev, curr, dist = queue.popleft()
            if curr == s:
                if dist < g:
                    return False
                continue
            if dist >= g:
                continue
            for nb in adj[curr]:
                if nb != prev and (curr, nb) not in visited:
                    visited[(curr, nb)] = dist + 1
                    queue.append((curr, nb, dist + 1))
    return True

def construct(k, g):
    deadline = time.time() + 140
    
    if k != 4 or g != 10:
        q = k
        d = 5
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
            r = (c - 5) % 4
            c0 = c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        
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
        n = len(order)
        return n, edges
    
    n_base, base_edges = _petersen_graph()
    best_result = None
    
    for m in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 20, 24, 30, 32, 40, 48, 60, 80]:
        if time.time() > deadline:
            break
        n_lift = 10 * m
        if n_lift > 500:
            continue
        
        rng = random.Random(12345 + m)
        for attempt in range(3000):
            if time.time() > deadline:
                break
            vol = {}
            for u, v in base_edges:
                vol[(u, v)] = rng.randint(0, m - 1)
            
            n_lift_actual, lift_edges = _lift_petersen(vol, m)
            
            if _check_girth_fast(lift_edges, n_lift_actual, 4, 10):
                if best_result is None or n_lift_actual < best_result[0]:
                    best_result = (n_lift_actual, lift_edges)
                break
    
    if best_result is not None:
        return best_result
    
    # Fallback: LUW
    q = 4
    d = 5
    table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
    add = lambda x, y: x ^ y
    sub = lambda x, y: x ^ y
    mul = lambda x, y: table[x][y]
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
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
    n = len(order)
    return n, edges
# EVOLVE-BLOCK-END
