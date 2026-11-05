import time
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    q = k
    
    # Helper to build LUW graph CD(d, q)
    def build_luw(d):
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
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
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

    def check_girth(n, edges, target_girth):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # BFS from each node
        # We only care if girth < target_girth
        # Max depth to search: target_girth // 2
        max_depth = target_girth // 2
        
        for s in range(n):
            # dist[v] = distance from s
            # parent[v] = previous node in BFS tree
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                if dist[u] >= max_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        # Found a back edge
                        # Cycle length = dist[u] + dist[v] + 1
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < target_girth:
                            return False
        return True

    # Seed's logic for d
    d_seed = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    # Try smaller d values first
    # Minimum d must be at least 3 for LUW to be well-defined in this context
    # Also d must be such that the graph can exist.
    # For g=9, try d=4, then d=5.
    
    candidates = []
    # We want small n, so small d.
    # But d must be large enough for girth.
    # Theoretical lower bound for d:
    # If girth is approx d+5 (odd) or d+4 (even),
    # we need d+4 >= g => d >= g-4.
    # So d_min = max(3, g-4).
    
    d_min = max(3, g - 4)
    
    for d in range(d_min, d_seed + 1):
        if time.time() > deadline:
            break
        n, e = build_luw(d)
        # Verify girth
        if check_girth(n, e, g):
            return n, e
            
    # Fallback to seed's d (which is guaranteed or highly likely valid)
    n, e = build_luw(d_seed)
    return n, e
