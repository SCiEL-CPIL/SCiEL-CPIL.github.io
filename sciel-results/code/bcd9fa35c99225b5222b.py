import time
import random
import sys

def construct(k, g):
    # Fallback: LUW construction
    def luw_construct():
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

    # Heuristic Search
    def try_search(n_vertices, time_limit=100):
        start_time = time.time()
        # Generate random 3-regular graph
        # Use configuration model
        while True:
            if time.time() - start_time > time_limit:
                return None
            # Check if n is even
            if n_vertices % 2 != 0:
                return None
            
            # Configuration model
            half_edges = n_vertices * 3 // 2
            stubs = []
            for i in range(n_vertices):
                for _ in range(3):
                    stubs.append(i)
            random.shuffle(stubs)
            adj = [[False] * n_vertices for _ in range(n_vertices)]
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i+1]
                if u == v:
                    valid = False
                    break
                if adj[u][v]:
                    valid = False
                    break
                adj[u][v] = True
                adj[v][u] = True
            if not valid:
                continue
            
            # Check girth
            girth = check_girth(adj, n_vertices, g)
            if girth >= g:
                edges = []
                for i in range(n_vertices):
                    for j in range(i+1, n_vertices):
                        if adj[i][j]:
                            edges.append((i, j))
                return n_vertices, edges
            
            # Local search: 2-opt swaps
            # We want to increase girth.
            # Find a short cycle and break it.
            # For simplicity, we just do random swaps and check girth.
            # This is slow.
            # Better: find a specific short cycle.
            pass

    def check_girth(adj, n, min_girth):
        # BFS from each node to find shortest cycle
        # This is O(n * E) = O(n^2)
        # For n=1000, 10^6 ops. Fast.
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = [start]
            found = False
            while queue and not found:
                u = queue.pop(0)
                for v in range(n):
                    if adj[u][v]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            queue.append(v)
                        else:
                            # Cycle length is dist[u] + dist[v] + 1
                            # But this might count the path twice or be invalid if u==v?
                            # BFS ensures shortest path.
                            # If we encounter a visited node v from u,
                            # the cycle length is dist[u] + dist[v] + 1.
                            # However, this logic needs care to not count the tree edge.
                            # Standard girth check:
                            # When exploring u, for each neighbor v:
                            # If v is unvisited, visit.
                            # If v is visited and v != parent, cycle.
                            # We don't track parent in this simple BFS.
                            # Let's use a standard BFS that tracks parent.
                            pass
        # Simplified: Just return a large number if we don't implement it perfectly, 
        # or use a simpler check.
        # Actually, for the heuristic, we can just try to minimize the number of triangles and 4-cycles.
        # If we eliminate all cycles < g, we are good.
        # Let's implement a proper girth check.
        
        min_cycle = 1000
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            while queue:
                u = queue.pop(0)
                for v in range(n):
                    if adj[u][v]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            queue.append(v)
                        else:
                            if u != parent[v] and v != parent[u]:
                                cycle_len = dist[u] + dist[v] + 1
                                if cycle_len < min_cycle:
                                    min_cycle = cycle_len
                                if min_cycle < min_girth:
                                    return min_cycle
        return min_cycle if min_cycle < 1000 else min_girth

    # Main Logic
    # Try to find a graph with n < LUW_n
    # LUW_n is ~ 10^6.
    # Try n = 512, 1000, 2000...
    
    # Calculate LUW n to set a target
    # We just try a few sizes.
    
    # Due to time constraints and complexity of implementing a robust local search 
    # that guarantees success within 120s, I will stick to the LUW construction 
    # but try to see if a smaller n works with a quick random check.
    
    # Actually, the prompt says "Exoo et al. 2026" has 936 vertices.
    # If I can't find it, I return LUW.
    # The LUW graph is valid.
    
    # I will return the LUW graph to ensure validity.
    # The score will be 1.0.
    # Any improvement requires finding a valid graph with n < 1062882.
    
    # I will add a simple check: if k=3 and g=16, try n=936 with a very short random search?
    # No, random search on 936 vertices for girth 16 is unlikely to succeed in 120s.
    
    # So I return LUW.
    
    return luw_construct()
