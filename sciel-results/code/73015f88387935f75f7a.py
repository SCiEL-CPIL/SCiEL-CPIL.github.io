import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 100  # 100 seconds budget for search
    
    # Fallback: LUW graph
    def luw_construct(k, g):
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

    # Voltage graph construction
    def check_girth(n, edges, g):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Check for cycles of length < g
        # Since graph is bipartite, check even lengths 2, 4, ..., g-1
        # Non-backtracking walks
        for u in range(n):
            # Generate walks
            # State: (current_node, prev_node, length)
            # We can use a set of visited (node, prev) pairs to avoid re-exploring?
            # No, we need to detect if we return to u.
            # Simple DFS/BFS up to depth g-1
            # For each start u, do BFS up to depth g-1
            # If we reach u again, we found a cycle.
            # But we must ensure the cycle is simple?
            # A non-backtracking walk that returns to u forms a cycle.
            # If the walk is non-backtracking, the cycle is simple.
            
            visited = set()
            queue = [(u, -1, 0)]
            while queue:
                node, prev, depth = queue.pop(0)
                if depth > 0 and node == u:
                    return False
                if depth >= g - 1:
                    continue
                for nb in adj[node]:
                    if nb == prev:
                        continue
                    if (node, nb) in visited:
                        continue
                    visited.add((node, nb))
                    queue.append((nb, node, depth + 1))
        return True

    def try_voltage_graph(base_edges, m, g):
        n_base = len(set([v for e in base_edges for v in e]))
        n = n_base * m
        edges = []
        # Assign random voltages
        voltages = {}
        for e in base_edges:
            u, v = e
            voltages[(u, v)] = random.randint(0, m - 1)
            voltages[(v, u)] = (-voltages[(u, v)]) % m
        
        for u in range(n_base):
            for v in [e[1] for e in base_edges if e[0] == u] + [e[0] for e in base_edges if e[1] == u]:
                if u == v: continue
                if (u, v) in voltages:
                    vu = voltages[(u, v)]
                    for i in range(m):
                        edges.append((u * m + i, v * m + (i + vu) % m))
        
        if check_girth(n, edges, g):
            return n, edges
        return None

    # Petersen Graph
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    
    # Dodecahedron Graph (20 vertices)
    # Correct edges for Dodecahedron
    dodeca_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0), # Top face? No, this is a 5-cycle.
        (5, 6), (6, 7), (7, 8), (8, 9), (9, 5), # Bottom face?
        (10, 11), (11, 12), (12, 13), (13, 14), (14, 10), # Middle?
        # Connections
        (0, 10), (1, 11), (2, 12), (3, 13), (4, 14),
        (5, 10), (6, 11), (7, 12), (8, 13), (9, 14),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    # This dodecahedron definition might have 4-cycles?
    # 0-1-11-10-0 is a 4-cycle.
    # Dodecahedron has girth 5.
    # So this edge list is wrong.
    # I will skip Dodecahedron if Petersen works.
    # Petersen has 10 vertices. m=51 gives n=510.
    
    best_result = None
    
    # Try Petersen with m from 51 to 60
    for m in range(51, 65):
        if time.time() > deadline:
            break
        for _ in range(5): # 5 trials per m
            if time.time() > deadline:
                break
            res = try_voltage_graph(petersen_edges, m, g)
            if res:
                best_result = res
                return res # Return immediately if found
    
    # If not found, return LUW
    return luw_construct(k, g)
