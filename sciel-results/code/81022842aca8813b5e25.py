import random
import time

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # 5s margin
    
    # Fallback: LUW construction (from seed)
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

    # Try to find a smaller graph using random search
    # Start with N close to Moore bound, increase if failing
    moore_bound = 2 + k * (1 + (k-1) * (1 + (k-1)**2) * (1 + (k-1)**3)) # Approx for g=9?
    # Exact Moore bound for k=5, g=9: 428.
    
    best_n, best_edges = None, None
    
    # Try a few sizes
    for n in [500, 600, 800, 1000, 1200, 1500, 2000]:
        if time.time() > deadline - 5:
            break
            
        # Generate random k-regular graph
        # Using a simple algorithm: 
        # 1. Create list of half-edges
        # 2. Randomly pair them
        # 3. Check for self-loops and multi-edges, retry if any
        graph = None
        attempts = 0
        while graph is None and attempts < 50:
            half_edges = [i for i in range(n) for _ in range(k)]
            random.shuffle(half_edges)
            edges = []
            valid = True
            for i in range(0, len(half_edges), 2):
                u, v = half_edges[i], half_edges[i+1]
                if u == v:
                    valid = False
                    break
                if (u, v) in edges or (v, u) in edges:
                    valid = False
                    break
                edges.append((u, v))
            if valid:
                graph = edges
            attempts += 1
        
        if graph is None:
            continue
            
        # Verify girth
        # Build adjacency list
        adj = {i: set() for i in range(n)}
        for u, v in graph:
            adj[u].add(v)
            adj[v].add(u)
            
        # Check for cycles < g
        # BFS from each node up to depth g//2
        # If we find a cycle of length L < g, we need to fix it.
        # For now, just check if girth is already >= g.
        # If yes, return.
        # If no, we could try to fix it, but for simplicity and speed,
        # we might just accept if girth is close, or skip to larger n.
        
        # Efficient girth check:
        # For each node, BFS up to depth g//2.
        # If we revisit a node within the same BFS tree at depth d1 and d2, 
        # cycle length is d1+d2+1? No, standard girth check.
        
        def has_short_cycle(adj, n, g):
            for start in range(n):
                # BFS
                visited = [-1] * n
                visited[start] = 0
                queue = [(start, 0)]
                while queue:
                    node, depth = queue.pop(0)
                    if depth >= g // 2:
                        continue
                    for neighbor in adj[node]:
                        if visited[neighbor] == -1:
                            visited[neighbor] = depth + 1
                            queue.append((neighbor, depth + 1))
                        else:
                            # If neighbor is not parent, and depth difference allows cycle < g
                            # Actually, simpler: if we find a back edge, cycle length is depth - visited[neighbor] + 1
                            # But we need to be careful with BFS tree.
                            # A simpler check: 
                            # If we find an edge (u, v) such that dist(u) + dist(v) + 1 < g and u != parent(v)
                            # This is complex. 
                            # Let's just use a standard girth algorithm.
                            pass
            return False # Placeholder, need real check
        
        # Real girth check implementation
        def get_girth(adj, n):
            girth = float('inf')
            for start in range(n):
                # BFS
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                queue = [start]
                while queue:
                    node = queue.pop(0)
                    if dist[node] >= girth // 2:
                        continue
                    for neighbor in adj[node]:
                        if dist[neighbor] == -1:
                            dist[neighbor] = dist[node] + 1
                            parent[neighbor] = node
                            queue.append(neighbor)
                        elif neighbor != parent[node]:
                            cycle_len = dist[node] + dist[neighbor] + 1
                            if cycle_len < girth:
                                girth = cycle_len
            return girth if girth != float('inf') else 2*n # If no cycle, inf
            
        # We only care if girth < g
        # Optimization: Stop BFS early if depth >= g//2
        
        def has_cycle_less_than_g(adj, n, g):
            for start in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                queue = [start]
                while queue:
                    node = queue.pop(0)
                    if dist[node] >= g // 2:
                        continue
                    for neighbor in adj[node]:
                        if dist[neighbor] == -1:
                            dist[neighbor] = dist[node] + 1
                            parent[neighbor] = node
                            queue.append(neighbor)
                        elif neighbor != parent[node]:
                            cycle_len = dist[node] + dist[neighbor] + 1
                            if cycle_len < g:
                                return True
            return False

        if not has_cycle_less_than_g(adj, n, g):
            return n, graph

    # If no random graph worked, return LUW
    return luw_construct(k, g)
