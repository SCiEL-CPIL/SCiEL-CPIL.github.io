import time
import math
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    def luw_construct(k, g):
        # ... (keep existing LUW code as fallback) ...
        q = k
        d = max(3, (g - 2) // 2)
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
            c0 = c - r
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
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))

        n = len(order)
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)

        min_cycle = float('inf')
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            while qi < len(queue):
                node = queue[qi]
                qi += 1
                for nb in adj[node]:
                    if nb == parent[node]:
                        continue
                    if dist[nb] != -1:
                        cycle_len = dist[node] + dist[nb] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if cycle_len < g:
                            return None
                    else:
                        dist[nb] = dist[node] + 1
                        parent[nb] = node
                        queue.append(nb)
            if min_cycle < g:
                return None

        return n, edges

    def check_girth(n, edges, g):
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        min_cycle = float('inf')
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            while qi < len(queue):
                node = queue[qi]
                qi += 1
                for nb in adj[node]:
                    if nb == parent[node]:
                        continue
                    if dist[nb] != -1:
                        cycle_len = dist[node] + dist[nb] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if cycle_len < g:
                            return False
                    else:
                        dist[nb] = dist[node] + 1
                        parent[nb] = node
                        queue.append(nb)
                if min_cycle < g:
                    return False
        return True

    def find_shortest_cycle(n, adj):
        # Returns length of shortest cycle
        min_cycle = float('inf')
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            while qi < len(queue):
                node = queue[qi]
                qi += 1
                for nb in adj[node]:
                    if nb == parent[node]:
                        continue
                    if dist[nb] != -1:
                        cycle_len = dist[node] + dist[nb] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                    else:
                        dist[nb] = dist[node] + 1
                        parent[nb] = node
                        queue.append(nb)
        return min_cycle

    if k != 3:
        result = luw_construct(k, g)
        if result is not None:
            return result
        return luw_construct(k, g)

    # Try Voltage Graph Construction over Petersen Graph
    # Petersen Graph: 10 vertices, 15 edges
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0), # Outer 5-cycle
        (5, 6), (6, 7), (7, 8), (8, 9), (9, 5), # Inner 5-cycle
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)  # Spokes
    ]
    
    best_result = None
    best_n = float('inf')

    # Try different group sizes m such that 10*m is close to 936
    # 936 / 10 = 93.6
    for m in [94, 92, 96, 90, 98, 100]:
        if time.time() > deadline - 20:
            break
            
        # Heuristic search for voltages
        # We want to avoid cycles of length < 16 in the cover graph.
        # A cycle in the cover graph corresponds to a closed walk in the base graph
        # with total voltage 0.
        
        # Initialize voltages randomly
        # To speed up, we can fix some voltages and search others?
        # Or just random restarts.
        
        found = False
        for restart in range(100): # 100 restarts
            if time.time() > deadline - 20:
                break
                
            voltages = [random.randint(0, m-1) for _ in range(len(petersen_edges))]
            
            # Build graph
            n = 10 * m
            edges = []
            for u in range(10):
                for v, vol in zip([e[1] for e in petersen_edges if e[0]==u], [vol for e, vol in zip(petersen_edges, voltages) if e[0]==u]):
                    # This logic is flawed, need to map edge index to voltage correctly
                    pass
            
            # Correct construction:
            # Map edge index to voltage
            # For each edge (u,v) with voltage vol:
            # Connect (u, i) to (v, i+vol) and (v, i) to (u, i-vol) ??
            # Standard voltage graph: edge (u,v) with voltage g connects (u, i) to (v, i+g).
            
            edges = []
            for i, (u, v) in enumerate(petersen_edges):
                vol = voltages[i]
                for j in range(m):
                    u_prime = (u, j)
                    v_prime = (v, (j + vol) % m)
                    # Add undirected edge
                    # Vertex id: u*m + j
                    edges.append((u*m + j, v*m + (j + vol) % m))
            
            # Check girth
            # Fast check: if n > 10000, skip? No, 940 is small.
            if check_girth(n, edges, g):
                best_result = (n, edges)
                best_n = n
                found = True
                break
            
            # If not found, try to repair?
            # For simplicity, just random restarts.
            # 100 restarts * 120s / 100 = 1.2s per restart.
            # Check girth on 940 vertices is fast.

    if best_result is not None:
        return best_result

    # Fallback to LUW
    result = luw_construct(k, g)
    if result is not None:
        return result
    return luw_construct(k, g)
