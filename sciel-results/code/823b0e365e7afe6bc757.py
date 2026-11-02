import time
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 118  # 32s safety margin before 150s kill
    
    def time_left():
        return deadline - time.time()
    
    def luw_construct(k, g, d_override=None):
        q = k
        if d_override is not None:
            d = d_override
        else:
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

    # Try LUW construction with default d
    best_n, best_edges = luw_construct(k, g)
    
    # Also try smaller d values to see if they give valid girth
    candidates = [(best_n, best_edges)]
    default_d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    for d_try in range(3, default_d):
        if time_left() < 5:
            break
        n_try, edges_try = luw_construct(k, g, d_override=d_try)
        candidates.append((n_try, edges_try))
    
    # Fast girth verification using non-backtracking walks
    def has_short_cycle(n, edges, g):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        max_depth = (g - 2) // 2
        
        for s in range(n):
            dist = {}
            queue = deque()
            for nb in adj[s]:
                dist[(nb, s)] = 1
                queue.append((nb, s))
            
            while queue:
                node, prev = queue.popleft()
                d_cur = dist[(node, prev)]
                if d_cur > max_depth:
                    continue
                for nb in adj[node]:
                    if nb == prev:
                        continue
                    if nb == s and d_cur + 1 < g:
                        return True
                    state = (nb, node)
                    if state not in dist:
                        dist[state] = d_cur + 1
                        queue.append(state)
        return False

    # Check candidates from smallest to largest
    candidates.sort(key=lambda x: x[0])
    for n_cand, edges_cand in candidates:
        if time_left() < 2:
            break
        if not has_short_cycle(n_cand, edges_cand, g):
            return n_cand, edges_cand
    
    # Fallback: return the default LUW result
    return best_n, best_edges
