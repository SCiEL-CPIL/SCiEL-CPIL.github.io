import time
import random
import sys
from collections import deque

def _luw_construct(k, g):
    # Reuse the logic from the current program
    q = k
    if (g - 5) % 2 != 0:
        d = max(3, g - 5)
    else:
        d = max(3, g - 4)
        
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q
        
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
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
                
    return len(order), edges

def _check_girth_lift(voltages, m):
    # Analytic 4-cycle check
    for i in range(4):
        for k in range(4):
            if i == k: continue
            for j in range(4):
                for l in range(4):
                    if j == l: continue
                    s = (voltages[i][j] + voltages[k][j] + voltages[l][k] + voltages[i][l]) % m
                    if s == 0:
                        return False # Has 4-cycle
            
    # BFS for 6 and 8 cycles
    n = 8 * m
    adj = [[] for _ in range(n)]
    for i in range(4):
        for j in range(4):
            w = voltages[i][j]
            for z in range(m):
                u = 0 * 4 * m + i * m + z
                v = 1 * 4 * m + j * m + (z + w) % m
                adj[u].append(v)
                adj[v].append(u)
                
    # Check for cycles of length 6 or 8
    # We can just run BFS up to depth 4 (for 8-cycles) from each node
    # and check if we return to start.
    
    for start in range(n):
        dq = deque()
        visited = set()
        for nb in adj[start]:
            dq.append((nb, start, 1))
            visited.add((nb, start))
            
        while dq:
            node, parent, depth = dq.popleft()
            if depth >= 4: # We are looking for cycles < 10.
                # Max depth for 8-cycle is 7 (edges) -> 8 nodes.
                # Path start->...->node->start.
                # If node is at depth d, and neighbor is start, cycle len d+1.
                # We want d+1 < 10 => d < 9.
                # So we need to go up to depth 8.
                # But we already checked 4-cycles analytically.
                # So we only need to check 6 and 8.
                # 6-cycle: depth 5.
                # 8-cycle: depth 7.
                # So we need to expand up to depth 7.
                pass
                
            for nb in adj[node]:
                if nb == parent:
                    continue
                if nb == start:
                    # Cycle length = depth + 1
                    # We know depth >= 1.
                    # If depth+1 == 4, it would have been caught by analytic check?
                    # Yes, analytic check catches all 4-cycles.
                    # So if depth+1 == 6 or 8, we return False.
                    if depth + 1 < 10:
                        return False
                    continue
                
                state = (nb, node)
                if state not in visited:
                    visited.add(state)
                    # We only need to go up to depth 7 (for 8-cycle)
                    if depth + 1 <= 7:
                        dq.append((nb, node, depth + 1))
                        
    return True

def _build_lifted_graph(voltages, m):
    n = 8 * m
    edges = []
    for i in range(4):
        for j in range(4):
            w = voltages[i][j]
            for z in range(m):
                u = 0 * 4 * m + i * m + z
                v = 1 * 4 * m + j * m + (z + w) % m
                edges.append((u, v))
    return n, edges

def construct(k, g):
    deadline = time.time() + 110.0
    
    if k != 4 or g != 10:
        return _luw_construct(k, g)
        
    best_n = float('inf')
    best_edges = None
    
    # Try m from 31 to 50
    for m in range(31, 51):
        if time.time() > deadline - 5.0:
            break
            
        max_tries = 2000
        for _ in range(max_tries):
            if time.time() > deadline - 2.0:
                break
                
            voltages = [[random.randint(0, m-1) for _ in range(4)] for _ in range(4)]
            
            if _check_girth_lift(voltages, m):
                n, edges = _build_lifted_graph(voltages, m)
                if n < best_n:
                    best_n = n
                    best_edges = edges
                    return n, edges
                    
    if best_edges is not None:
        return best_n, best_edges
        
    return _luw_construct(k, g)
