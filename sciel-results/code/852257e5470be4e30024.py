import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0

    # Fallback: LUW construction (2048 vertices)
    q = 4
    d = 5
    table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
    def gf_add(x, y): return x ^ y
    def gf_sub(x, y): return x ^ y
    def gf_mul(x, y): return table[x][y]
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    def neighbours_luw(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = gf_add(x[c], gf_mul(y[a], x[b]))
                else:
                    y[c] = gf_sub(x[c], gf_mul(x[a], y[b]))
            yield 1 - side, tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours_luw(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    best_n = len(order)
    best_edges = edges

    # Try to find 320-vertex graph (Foster Cage) via K4,4 lift
    if k == 4 and g == 10:
        M = 40
        n_base = 8
        # K4,4 edges
        base_edges = [(i, j) for i in range(4) for j in range(4, 8)]
        base_adj = [[] for _ in range(n_base)]
        for u, v in base_edges:
            base_adj[u].append(v)
            base_adj[v].append(u)
            
        def check_girth_lift(vol_dict, M, g):
            # Check for cycles of length < g
            # State: (prev, curr, depth, vol_sum)
            # Start from each vertex s
            for s in range(n_base):
                visited = set()
                queue = []
                for nb in base_adj[s]:
                    vol = vol_dict.get((s, nb), 0)
                    state = (s, nb, 1, vol % M)
                    queue.append(state)
                    visited.add(state)
                
                while queue:
                    new_queue = []
                    for prev, curr, depth, vol_sum in queue:
                        if depth >= g - 1:
                            continue
                        for nb in base_adj[curr]:
                            if nb == prev:
                                continue
                            vol = vol_dict.get((curr, nb), 0)
                            new_vol = (vol_sum + vol) % M
                            if nb == s and depth + 1 < g and new_vol == 0:
                                return False
                            state = (curr, nb, depth + 1, new_vol)
                            if state not in visited:
                                visited.add(state)
                                new_queue.append(state)
                    queue = new_queue
            return True

        # Search for valid voltage assignment
        random.seed(42)
        found = False
        max_attempts = 1000
        for _ in range(max_attempts):
            if time.time() > deadline - 20.0:
                break
            voltages = [random.randint(0, M-1) for _ in base_edges]
            vol_dict = {}
            for (u, v), vol in zip(base_edges, voltages):
                vol_dict[(u, v)] = vol
                vol_dict[(v, u)] = (-vol) % M
            
            if check_girth_lift(vol_dict, M, g):
                # Build graph
                n_total = n_base * M
                adj = [[] for _ in range(n_total)]
                for (u, v), vol in zip(base_edges, voltages):
                    for w in range(M):
                        a = u * M + w
                        b = v * M + ((w + vol) % M)
                        adj[a].append(b)
                        adj[b].append(a)
                
                # Verify
                if all(len(adj[i]) == 4 for i in range(n_total)):
                    edge_set = set()
                    for i in range(n_total):
                        for j in adj[i]:
                            if i < j:
                                edge_set.add((i, j))
                    if len(edge_set) == 4 * n_total // 2:
                        best_n = n_total
                        best_edges = list(edge_set)
                        found = True
                        break
    return best_n, best_edges
