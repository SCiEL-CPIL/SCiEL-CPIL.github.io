import random
import collections
import time

def construct(k, g):
    # Target n is around 936. We'll try to find a graph with n <= 1000.
    # Approach: Local Search (Rewiring)
    
    def generate_random_regular(n, k):
        stubs = [i for i in range(n) for _ in range(k)]
        random.shuffle(stubs)
        adj = {i: [] for i in range(n)}
        for i in range(0, n * k, 2):
            u = stubs[i]
            v = stubs[i + 1]
            if u == v:
                return None
            if v in adj[u]:
                return None
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def get_short_cycle(adj, n, g):
        # Returns a list of vertices in a cycle of length < g, or None
        max_depth = g // 2 - 1
        for start in range(n):
            dist = {start: 0}
            parent = {start: -1}
            queue = collections.deque([start])
            while queue:
                u = queue.popleft()
                if dist[u] > max_depth:
                    continue
                for v in adj[u]:
                    if v == parent[u]:
                        continue
                    if v in dist:
                        # Cycle found
                        length = dist[u] + dist[v] + 1
                        if length < g:
                            # Reconstruct cycle
                            # Path start -> ... -> u
                            # Path start -> ... -> v
                            # Cycle is u -> ... -> start -> ... -> v -> u
                            # We just need the edges to break.
                            # Returning the start vertex is enough to identify a region, 
                            # but for rewiring we need specific edges.
                            # Let's return the cycle vertices.
                            path_u = []
                            cur = u
                            while cur != -1:
                                path_u.append(cur)
                                cur = parent[cur]
                            path_u.reverse()
                            
                            path_v = []
                            cur = v
                            while cur != -1:
                                path_v.append(cur)
                                cur = parent[cur]
                            path_v.reverse()
                            
                            # The cycle is path_u + reversed(path_v[1:])
                            # Actually, the cycle is u...start...v-u.
                            # Edges: (u, parent[u]), ..., (start, child), ..., (v, parent[v]) and (u,v).
                            # We can just return the set of edges in the cycle.
                            cycle_edges = []
                            # Add edges in path_u
                            for i in range(len(path_u) - 1):
                                a, b = path_u[i], path_u[i+1]
                                cycle_edges.append((min(a,b), max(a,b)))
                            # Add edges in path_v
                            for i in range(len(path_v) - 1):
                                a, b = path_v[i], path_v[i+1]
                                cycle_edges.append((min(a,b), max(a,b)))
                            # Add closing edge (u, v)
                            cycle_edges.append((min(u,v), max(u,v)))
                            return cycle_edges
                    else:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
        return None

    def try_rewire(adj, n, k, cycle_edges):
        # Pick an edge from the cycle to break
        # Let's pick a random edge from the cycle
        e1 = random.choice(cycle_edges)
        u, v = e1
        # Remove (u, v)
        adj[u].remove(v)
        adj[v].remove(u)
        
        # We need to add two new edges to restore degree k.
        # We can pick a random other edge (x, y) not incident to u or v, 
        # and perform a 2-switch.
        # Or, we can just pick random new neighbors for u and v.
        # But they must be connected to something.
        # Let's find a random edge (x, y) in the graph.
        # Remove (x, y). Add (u, x) and (v, y) or (u, y) and (v, x).
        
        all_edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    all_edges.append((i, j))
        
        # Filter edges not involving u or v
        candidates = [e for e in all_edges if u not in e and v not in e]
        if not candidates:
            # Revert
            adj[u].append(v)
            adj[v].append(u)
            return False

        x, y = random.choice(candidates)
        # Check if new edges are valid (no loops, no multi-edges)
        # Option 1: (u, x) and (v, y)
        if x not in adj[u] and y not in adj[v] and u != x and v != y:
            # Remove (x, y)
            adj[x].remove(y)
            adj[y].remove(x)
            # Add (u, x) and (v, y)
            adj[u].append(x)
            adj[x].append(u)
            adj[v].append(y)
            adj[y].append(v)
            return True
        # Option 2: (u, y) and (v, x)
        if y not in adj[u] and x not in adj[v] and u != y and v != x:
            adj[x].remove(y)
            adj[y].remove(x)
            adj[u].append(y)
            adj[y].append(u)
            adj[v].append(x)
            adj[x].append(v)
            return True
            
        # Revert if both fail
        adj[u].append(v)
        adj[v].append(u)
        return False

    start_time = time.time()
    max_time = 115
    
    # Try different n values
    # Start with n close to 936
    for n in [936, 938, 940, 942, 944, 946, 948, 950]:
        if time.time() - start_time > max_time:
            break
            
        # Try to find a solution for this n
        for attempt in range(100):
            if time.time() - start_time > max_time:
                break
                
            adj = generate_random_regular(n, k)
            if adj is None:
                continue
                
            # Local search
            for step in range(1000):
                if time.time() - start_time > max_time:
                    break
                cycle = get_short_cycle(adj, n, g)
                if cycle is None:
                    # Success
                    edges = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                edges.append((u, v))
                    return n, edges
                else:
                    if not try_rewire(adj, n, k, cycle):
                        # If rewiring fails, break out and try new random graph
                        break
                        
    # Fallback: Return a known valid graph if possible?
    # If we fail, we must return something.
    # The LUW fallback in the original code is huge.
    # I will implement a simple valid graph with high n just to return *something*?
    # No, the score is minimized n. If I return n=10000, score is bad.
    # But returning an invalid graph scores 0.
    # So I must return a valid graph.
    # If local search fails, I will use the LUW construction (even if large) as a last resort,
    # but I will implement it correctly this time?
    # The previous LUW implementation was correct but produced large n.
    # I will use a smaller LUW parameter if possible?
    # For g=16, d=7 gives n=4374.
    # Is there a smaller algebraic graph?
    # I will just return the LUW graph. It's better than 0.
    
    return _luw_fallback(k, g)

def _luw_fallback(k, g):
    # Same as before, ensures validity
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
