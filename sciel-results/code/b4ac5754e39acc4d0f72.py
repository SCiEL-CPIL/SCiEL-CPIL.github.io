import time
import sys
import random
import numpy as np
from collections import defaultdict

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115
    
    # 1. Generate the seed's LUW graph as a baseline
    n_luw, edges_luw = _construct_luw(k, g)
    best_n = n_luw
    best_edges = edges_luw
    
    # 2. Try Voltage Lift Construction with 3-regular base graphs
    # Petersen graph: 10 vertices, girth 5
    # Heawood graph: 14 vertices, girth 6
    
    # Define Petersen graph
    # Vertices 0..9
    # Edges: (0,1), (1,2), (2,3), (3,4), (4,0) [outer]
    # (5,7), (7,9), (9,6), (6,8), (8,5) [inner]
    # (0,5), (1,6), (2,7), (3,8), (4,9) [spokes]
    petersen_edges = [(0,1), (1,2), (2,3), (3,4), (4,0),
                      (5,7), (7,9), (9,6), (6,8), (8,5),
                      (0,5), (1,6), (2,7), (3,8), (4,9)]
    
    # Try lifting Petersen with small m
    for m in [4, 6, 8, 10]:
        if time.time() > deadline - 20:
            break
        if 10 * m >= best_n:
            break
            
        # Random search for labels
        # Petersen has 15 edges
        # We need to find labels in Z_m such that no 5-cycle sums to 0 mod m
        # (and no other short cycles, but girth is 5)
        # Actually, we just need to check the resulting graph
        
        for attempt in range(100):
            if time.time() > deadline - 20:
                break
            labels = [random.randint(0, m-1) for _ in range(15)]
            edges = _construct_lift_petersen(m, labels)
            if _has_girth_at_least_fast(10*m, edges, g):
                return 10*m, edges
                
    return best_n, best_edges

def _construct_lift_petersen(m, labels):
    n = 10 * m
    edges = set()
    petersen_edges = [(0,1), (1,2), (2,3), (3,4), (4,0),
                      (5,7), (7,9), (9,6), (6,8), (8,5),
                      (0,5), (1,6), (2,7), (3,8), (4,9)]
    
    for i, (u, v) in enumerate(petersen_edges):
        label = labels[i]
        for g in range(m):
            u_id = u * m + g
            v_id = v * m + (g + label) % m
            edges.add((u_id, v_id))
            
    return list(edges)

def _construct_luw(k, g):
    # Lazy-evaluated LUW construction from seed
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

def _has_girth_at_least_fast(n, edges, g):
    return _has_girth_at_least_bfs(n, edges, g)

def _has_girth_at_least_bfs(n, edges, g):
    """Standard BFS girth check, optimized with early exit."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
                else:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True
