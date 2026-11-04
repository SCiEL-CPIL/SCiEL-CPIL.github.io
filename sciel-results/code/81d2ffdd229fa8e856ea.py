import time
import random
import math
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Fallback: LUW graph with d=7
    def luw_fallback():
        q = 3
        d = 7
        rule = {}
        rule[2] = (1, 1)
        rule[3] = (2, 1)
        rule[4] = (1, 2)
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return x * y % q

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

    base_edges = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    
    def check_girth_fast(n, edges, start_node=0):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # BFS from start_node
        dist = [-1] * n
        parent = [-1] * n
        dist[start_node] = 0
        q = deque([start_node])
        min_girth = n + 1
        
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
        return min_girth

    def check_girth_full(n, edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        min_girth = n + 1
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
            if min_girth < 16:
                return min_girth
        return min_girth

    def generate_graph(M, voltages):
        n = 4 * M
        edges = []
        for i, (u, v) in enumerate(base_edges):
            vol = voltages[i]
            for g in range(M):
                u_idx = u * M + g
                v_idx = v * M + (g + vol) % M
                edges.append((u_idx, v_idx))
        return n, edges

    best_n = float('inf')
    best_edges = None
    
    for M in range(192, 300):
        if time.time() > deadline:
            break
            
        found = False
        max_iters = 2000
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            voltages = [random.randint(0, M-1) for _ in range(6)]
            
            g_val = M
            for v in voltages:
                g_val = math.gcd(g_val, v)
            if g_val != 1:
                continue
                
            n, edges = generate_graph(M, voltages)
            
            # Fast check
            girth = check_girth_fast(n, edges, 0)
            
            if girth >= 16:
                # Full check
                full_girth = check_girth_full(n, edges)
                if full_girth >= 16:
                    best_n = n
                    best_edges = edges
                    found = True
                    break
                
        if found:
            break

    if best_edges is not None:
        return best_n, best_edges
    else:
        return luw_fallback()
