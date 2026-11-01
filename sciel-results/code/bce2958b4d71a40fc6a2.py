import time
import random
import sys
import math

def construct(k, g):
    # Fallback: Lazebnik-Ustimenko-Woldar graph
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

    # Moore bound calculation
    D = (g - 2) // 2
    moore_bound = 1 + k * sum((k - 1) ** i for i in range(D))
    
    deadline = time.time() + 110
    
    # Target n=320 for k=4, g=10
    target_n = 320 if k == 4 and g == 10 else moore_bound
    
    # Helper to check girth
    def has_girth_at_least(n, edges, g):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Check for simple graph
        edge_set = set()
        for u, v in edges:
            if u == v:
                return False
            if (v, u) in edge_set:
                return False
            edge_set.add((u, v))
        
        # BFS from each vertex
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                if dist[u] >= g // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    else:
                        if v != parent[u] and u != parent[v]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                return False
        return True

    # Helper to count short cycles (for optimization)
    def count_short_cycles(n, adj, max_len):
        # This is expensive, so we use a proxy or limited BFS
        # For now, just check if girth >= g
        # A better proxy: number of cycles of length 3, 4, 5
        count = 0
        # 3-cycles
        for u in range(n):
            for i in range(len(adj[u])):
                v = adj[u][i]
                if v <= u: continue
                for w in adj[v]:
                    if w > v and w in adj[u]:
                        count += 1
        # 4-cycles
        for u in range(n):
            for v in adj[u]:
                if v <= u: continue
                for w in adj[v]:
                    if w <= v: continue
                    for x in adj[w]:
                        if x > w and x == u:
                            count += 1
        return count

    # Simulated Annealing for 4-regular graph
    def optimize_girth(n, k, g, time_limit):
        # Generate initial random 4-regular graph
        def random_regular_graph(n, k):
            while True:
                stubs = []
                for i in range(n):
                    for _ in range(k):
                        stubs.append(i)
                random.shuffle(stubs)
                edges = []
                valid = True
                seen = set()
                for i in range(0, len(stubs), 2):
                    u = stubs[i]
                    v = stubs[i+1]
                    if u == v:
                        valid = False
                        break
                    if (u, v) in seen or (v, u) in seen:
                        valid = False
                        break
                    seen.add((u, v))
                    edges.append((u, v))
                if valid:
                    return edges

        edges = random_regular_graph(n, k)
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Map edge to index
        edge_map = {}
        for i, (u, v) in enumerate(edges):
            edge_map[(u, v)] = i
            edge_map[(v, u)] = i

        def get_cost():
            # Count cycles of length < g
            # Simplified: just check if has_girth_at_least is false, and if so, return 1, else 0?
            # No, we need a gradient.
            # Let's count 3, 4, 5, 6 cycles roughly.
            # This is too slow to do every step.
            # Instead, we will just try to find a solution and check.
            # If we can't, we return None.
            return None

        # Actually, for n=320, local search might be too slow to converge to girth 10.
        # I will just run a few random restarts and check.
        # If I don't find it, I return LUW.
        
        start_time = time.time()
        attempts = 0
        while time.time() - start_time < time_limit:
            edges = random_regular_graph(n, k)
            if has_girth_at_least(n, edges, g):
                return n, edges
            attempts += 1
            if attempts % 100 == 0:
                # Maybe do some local search here?
                # For now, just random restarts.
                pass
                
        return None

    # Try random search for target_n
    if k == 4 and g == 10:
        result = optimize_girth(320, 4, 10, 30)
        if result:
            return result
            
    # Fallback to LUW
    return luw_construct(k, g)
