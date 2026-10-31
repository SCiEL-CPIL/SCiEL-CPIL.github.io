import time
from collections import deque

def construct(k, g):
    deadline = time.time() + 110  # 10s safety margin
    
    # General fallback: LUW construction
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

    # For the hard case k=3, g=16, use a structured approach
    if k == 3 and g == 16:
        # Strategy: Build a 3-regular graph from a "cage-like" base using
        # a product/twisted construction, then verify girth.
        # 
        # The smallest known (3,16)-cage has 936 vertices.
        # The LUW construction gives ~350k vertices. We need something much smaller.
        #
        # Approach: Use a known small (3,16)-cage if we can hardcode it,
        # or generate via a smarter parameterized family.
        #
        # Let's try the "Andreas" or "Goedgebeur" style constructions:
        # These use a base of 3-cycles connected in a tree-like fashion
        # to achieve high girth with moderate vertex count.
        #
        # Actually, let's try a different approach: 
        # Start with the Petersen graph (3-regular, girth 5, 10 vertices)
        # and apply a "lift" or "blow-up" to increase girth.
        # 
        # A 3-regular graph with girth 16 must have at least the Moore bound:
        # n >= 1 + 3*(1 + 3 + 3^2 + ... + 3^(g/2 - 1)) / (g/2) ... 
        # Actually, for girth 16, the Moore bound is:
        # n >= 1 + 3 * sum_{i=0}^{6} 3^i = 1 + 3 * (3^7 - 1)/2 = 1 + 3 * 1093 = 3280
        # Wait, that's for girth 2*7 = 14. For girth 16, it's radius 8:
        # n >= 1 + 3 * sum_{i=0}^{7} 3^i = 1 + 3 * (3^8 - 1)/2 = 1 + 3 * 3280 = 9841
        # Hmm, but the known smallest is 936, so the Moore bound is not tight.
        # 
        # Let me reconsider: the Moore bound for (k,g) with even g=2r is:
        # n >= k * (k-1)^{r-1} / (r * (k-1) - 1) ... this is getting complicated.
        # 
        # Let's just try to find a valid graph via a different construction.
        #
        # Try: 3-regular graph from a circulant-like structure.
        # A (3,16)-cage is known to have 936 vertices. Let's try to construct
        # one using a "triple product" or "wreath" construction.
        
        # Alternative: Use the known fact that the (3,16)-cage can be constructed
        # as a "tree of triangles" or similar. Let's try a simpler approach:
        # Generate a random 3-regular graph and check girth, keeping the smallest
        # valid one found within time.
        
        import random
        random.seed(42)
        
        def find_shortest_cycle(adj, n, max_girth_check):
            """Find the shortest cycle in the graph using BFS from each vertex."""
            best = float('inf')
            for start in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                q = deque([start])
                while q:
                    u = q.popleft()
                    if dist[u] >= max_girth_check // 2:
                        continue
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            q.append(v)
                        elif v != parent[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < best:
                                best = cycle_len
            return best if best < float('inf') else float('inf')
        
        def generate_random_3_regular(n):
            """Generate a random 3-regular graph on n vertices."""
            if n % 2 != 0:
                return None
            # Use configuration model
            stubs = [i for i in range(n) for _ in range(3)]
            random.shuffle(stubs)
            edges = []
            for i in range(0, n * 3, 2):
                u, v = stubs[i], stubs[i + 1]
                if u == v:
                    return None  # Loop, retry
                if any((u == a and v == b) or (u == b and v == a) for a, b in edges):
                    return None  # Multi-edge, retry
                edges.append((u, v))
            
            # Check if it's connected
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            visited = [False] * n
            visited[0] = True
            q = deque([0])
            count = 1
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        count += 1
                        q.append(v)
            if count != n:
                return None  # Disconnected
            
            return edges
        
        def check_girth(adj, n, required_girth):
            """Check if girth is at least required_girth."""
            # BFS from each vertex up to depth required_girth // 2
            max_depth = required_girth // 2
            for start in range(n):
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                q = deque([start])
                while q:
                    u = q.popleft()
                    if dist[u] >= max_depth:
                        continue
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            q.append(v)
                        elif v != parent[u]:
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < required_girth:
                                return False
            return True
        
        # Try random 3-regular graphs of increasing size
        best_n = float('inf')
        best_edges = None
        
        # Start from a reasonable lower bound
        # The (3,16)-cage is known to have 936 vertices
        for n_try in range(936, 1200, 2):
            if time.time() > deadline - 10:
                break
            
            found = False
            for attempt in range(200):
                if time.time() > deadline - 10:
                    break
                edges = generate_random_3_regular(n_try)
                if edges is None:
                    continue
                
                adj = [[] for _ in range(n_try)]
                for u, v in edges:
                    adj[u].append(v)
                    adj[v].append(u)
                
                if check_girth(adj, n_try, g):
                    best_n = n_try
                    best_edges = edges
                    found = True
                    break
            
            if found:
                break
        
        if best_edges is not None:
            return best_n, best_edges
        
        # Fallback to LUW
        return luw_construct(k, g)
    
    # For other (k, g), use LUW
    return luw_construct(k, g)
