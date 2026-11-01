"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    deadline = time.time() + 110  # safety margin before 120s kill
    rng = random.Random(12345)

    def moore_bound(k, g):
        # Moore bound for k-regular graph of girth g
        if g % 2 == 0:
            half = g // 2
            # n >= k * (k-1)^(half-1) + (k-1)^(half-1) ... actually:
            # For even girth g=2r: n >= k * sum_{i=0}^{r-1} (k-1)^i
            total = 0
            term = 1
            for i in range(g // 2):
                total += term
                term *= (k - 1)
            return k * total
        else:
            # For odd girth g=2r+1: n >= 1 + k * sum_{i=0}^{r} (k-1)^i
            total = 1
            term = 1
            for i in range(g // 2 + 1):
                total += k * term if i == 0 else 0
                # Let me redo this properly
            # n >= 1 + k * (1 + (k-1) + (k-1)^2 + ... + (k-1)^r) where r = (g-1)/2
            r = (g - 1) // 2
            s = 0
            term = 1
            for i in range(r + 1):
                s += term
                term *= (k - 1)
            return 1 + k * s

    min_n = moore_bound(k, g)

    def check_girth(n, edges):
        """Check girth >= g using BFS from each vertex. Returns True if valid."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        # Check degree
        for i in range(n):
            if len(adj[i]) != k:
                return False
        # Check no short cycles using BFS from each vertex
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        # Found a cycle of length dist[u] + dist[v] + 1
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
        return True

    def check_girth_fast(n, edges):
        """Faster girth check: only need to check up to depth g//2 from each vertex."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        for i in range(n):
            if len(adj[i]) != k:
                return False
        max_depth = g // 2
        for start in range(n):
            # BFS up to max_depth, tracking parent to avoid backtracking
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            found_short = False
            while queue:
                u = queue.popleft()
                if dist[u] >= max_depth:
                    continue
                for v in adj[u]:
                    if v == parent[u]:
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        if dist[v] < max_depth:
                            queue.append(v)
                    else:
                        # This edge (u,v) where both are visited and v != parent[u]
                        # forms a cycle. The cycle length is dist[u] + dist[v] + 1
                        # but only if the path from start to u and start to v are disjoint
                        # Actually for girth checking, we just need: if dist[u] + dist[v] + 1 < g
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
        return True

    def build_random_girth_graph(n_target, k, g, seed=42):
        """
        Build a k-regular graph on n_target vertices with girth >= g
        using a randomized approach: start with a tree, then close it up.
        """
        rng_local = random.Random(seed)
        
        # Start with a k-regular tree structure (BFS layers)
        # For girth g, we need at least depth g//2 from any vertex without cycles
        # Strategy: build a "tree-like" graph and then add edges carefully
        
        # Actually, let's use a different approach:
        # Build the graph as a union of cycles or use a random regular graph
        # and check girth. For small n this might work.
        
        # Better: use the "greedy edge addition" method
        # Start with n vertices, all degree 0
        # Add edges one by one, ensuring:
        # 1. No vertex exceeds degree k
        # 2. No cycle shorter than g is formed
        # At the end, if all vertices have degree k, we're done
        
        n = n_target
        degree = [0] * n
        adj = [set() for _ in range(n)]
        edges = []
        
        def would_create_short_cycle(u, v, max_depth):
            """Check if adding edge (u,v) creates a cycle shorter than g."""
            # BFS from u and v simultaneously up to depth (g-1)//2
            # If they meet, the cycle length is dist_u + dist_v + 1
            depth_limit = (g - 1) // 2
            
            # BFS from u
            dist_u = {u: 0}
            queue_u = deque([(u, 0)])
            while queue_u:
                node, d = queue_u.popleft()
                if d >= depth_limit:
                    continue
                for nb in adj[node]:
                    if nb == v:  # we're about to add this edge
                        continue
                    if nb not in dist_u:
                        dist_u[nb] = d + 1
                        queue_u.append((nb, d + 1))
            
            # BFS from v
            dist_v = {v: 0}
            queue_v = deque([(v, 0)])
            while queue_v:
                node, d = queue_v.popleft()
                if d >= depth_limit:
                    continue
                for nb in adj[node]:
                    if nb == u:
                        continue
                    if nb not in dist_v:
                        dist_v[nb] = d + 1
                        queue_v.append((nb, d + 1))
            
            # Check if any vertex is reachable from both u and v
            # with dist_u[x] + dist_v[x] + 1 < g
            for x in dist_u:
                if x in dist_v:
                    cycle_len = dist_u[x] + dist_v[x] + 1
                    if cycle_len < g:
                        return True
            return False
        
        # This is too slow for large n. Let's use a different strategy.
        # For k=3, g=16, we need ~936 vertices.
        # Let's try: generate a random 3-regular graph and check girth.
        # If girth < 16, modify it.
        
        return None  # placeholder

    def build_cage_graph(k, g, max_time=100):
        """
        Try to build a k-regular graph of girth g with minimal vertices.
        Uses multiple strategies.
        """
        start_time = time.time()
        best_n = None
        best_edges = None
        
        # Strategy 1: Use known constructions
        # For k=3, g=16: known best is 936 vertices
        # For k=3, g=8: 30 vertices (Tutte-Coxeter)
        # For k=3, g=10: 70 vertices
        # For k=3, g=12: 126 vertices  
        # For k=3, g=14: 240 vertices
        
        # Let's try the Lazebnik-Ustimenko-Woldar construction which is in the seed
        # but try to optimize it or use a different parameterization
        
        # Actually, let's just use a simple approach:
        # Try to build a random k-regular graph with increasing n until girth >= g
        
        # For efficiency, let's use the fact that for k=3, g=16,
        # the answer is known to be 936. Let's try to construct it.
        
        # Use a random 3-regular graph approach:
        # 1. Start with n vertices
        # 2. Generate a random 3-regular graph
        # 3. Check girth
        # 4. If girth < g, try again or modify
        
        # But this is very slow for large n. Let's use a smarter approach.
        
        # Strategy: Use the "tree growth" method
        # Build a graph that is locally a tree up to depth (g-1)//2
        # This means no cycles of length < g
        
        # For k=3, g=16: depth = 7
        # A 3-regular tree of depth 7 has 1 + 3*(1+3+9+27+81+243+729) = 1 + 3*1093 = 3280 vertices
        # But we can share vertices at the boundary to reduce the count
        
        # Let's try a different approach: use the Wagner graph or other known cages
        # and generalize
        
        # For now, let's use a simple randomized approach with time limit
        moore = moore_bound(k, g)
        
        # Try n values from moore_bound upward
        # But for k=3, g=16, moore_bound is about 512, and we need 936
        # So we need to search a range
        
        # Given time constraints, let's try a few specific n values
        # and use random regular graph generation
        
        for trial_n in range(moore, moore + 500, 2):  # even numbers for k-regular
            if time.time() - start_time > max_time:
                break
            if trial_n * k % 2 != 0:
                continue
            
            # Try to build a random k-regular graph on trial_n vertices
            # with girth >= g
            success = False
            for attempt in range(5):
                if time.time() - start_time > max_time:
                    break
                
                edges = random_k_regular_graph(trial_n, k, rng_local)
                if edges is None:
                    continue
                
                # Quick girth check
                if check_girth_fast(trial_n, edges):
                    best_n = trial_n
                    best_edges = edges
                    success = True
                    break
            
            if success:
                break
        
        return best_n, best_edges
    
    def random_k_regular_graph(n, k, rng):
        """Generate a random k-regular graph on n vertices."""
        if n * k % 2 != 0:
            return None
        if n < k or k >= n:
            return None
        
        # Configuration model
        half_edges = n * k // 2
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        
        rng.shuffle(stubs)
        
        # Pair up stubs
        edges = set()
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                return None  # loop
            if (min(u,v), max(u,v)) in edges:
                return None  # multi-edge
            edges.add((min(u,v), max(u,v)))
        
        # Verify it's k-regular
        degree = [0] * n
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1
        if any(d != k for d in degree):
            return None
        
        return list(edges)
    
    # Run the construction
    result_n, result_edges = build_cage_graph(k, g, max_time=100)
    
    if result_n is not None and result_edges is not None:
        return result_n, result_edges
    
    # Fallback: use the Lazebnik-Ustimenko-Woldar construction from the seed
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

# EVOLVE-BLOCK-END
