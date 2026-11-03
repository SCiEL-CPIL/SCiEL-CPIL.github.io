import time
import random
import numpy as np

def construct(k, g):
    # Deadline guard
    start_time = time.time()
    timeout = 110.0  # 10s margin
    
    # If not the specific case, or if we fail, fallback to LUW
    def luw_fallback():
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

    # Specific optimization for k=3, g=16
    if k == 3 and g == 16:
        # Try to find a smaller graph via random search
        # Target N values
        target_ns = [512, 600, 800, 1000, 1200]
        
        def check_girth(adj, n, g):
            # Check if girth >= g
            # BFS from each node up to depth g//2 - 1
            depth_limit = g // 2 - 1
            for start in range(n):
                visited = [-1] * n
                parent = [-1] * n
                queue = [(start, 0)]
                visited[start] = 0
                while queue:
                    node, dist = queue.pop(0)
                    if dist >= depth_limit:
                        continue
                    for nei in adj[node]:
                        if visited[nei] == -1:
                            visited[nei] = dist + 1
                            parent[nei] = node
                            queue.append((nei, dist + 1))
                        elif parent[node] != nei and parent[nei] != node:
                            # Found a cycle
                            # dist(node) + dist(nei) + 1 is the cycle length
                            # We need to be careful with the tree structure
                            # If we find a back edge, cycle length is dist[node] + dist[nei] + 1
                            # But we stop at depth_limit, so we might miss long cycles?
                            # No, we want to find SHORT cycles.
                            # If a cycle is shorter than g, it must be found within depth g/2.
                            # Actually, if g is even, shortest cycle is 2*L.
                            # We search up to L-1.
                            # If we find a connection between two nodes in the BFS tree
                            # that are not parent-child, we have a cycle.
                            # The length is dist[node] + dist[nei] + 1.
                            # We need this to be >= g.
                            # If dist[node] + dist[nei] + 1 < g, return False
                            cycle_len = dist + visited[nei] + 1
                            if cycle_len < g:
                                return False
            return True

        def generate_random_3_regular(n):
            # Configuration model with rejection
            # Stubs: 3n
            stubs = []
            for i in range(n):
                stubs.extend([i, i, i])
            
            while True:
                random.shuffle(stubs)
                adj = [[] for _ in range(n)]
                valid = True
                for i in range(0, 3*n, 2):
                    u = stubs[i]
                    v = stubs[i+1]
                    if u == v:
                        valid = False
                        break
                    if v in adj[u]:
                        valid = False
                        break
                    adj[u].append(v)
                    adj[v].append(u)
                if valid:
                    return adj

        for n in target_ns:
            # Time budget per n
            t_start = time.time()
            t_limit = 15.0
            attempts = 0
            while time.time() - t_start < t_limit:
                adj = generate_random_3_regular(n)
                if check_girth(adj, n, g):
                    edges = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                edges.append((u, v))
                    return n, edges
                attempts += 1
                if time.time() > start_time + timeout:
                    break
            
            if time.time() > start_time + timeout:
                break

    return luw_fallback()
