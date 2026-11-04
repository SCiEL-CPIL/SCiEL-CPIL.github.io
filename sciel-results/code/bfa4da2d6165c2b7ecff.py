import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Moore Bound Calculation
    r = g // 2
    if g % 2 == 0:
        if k == 2:
            n_min = g
        else:
            # 1 + k * sum_{i=0}^{r-2} (k-1)^i
            n_min = 1 + k * sum((k-1)**i for i in range(r-1))
    else:
        if k == 2:
            n_min = g
        else:
            n_min = 1 + k * sum((k-1)**i for i in range(r-1))
    
    # Fallback: Lazebnik-Ustimenko-Woldar
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
            r_val, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r_val]

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

    # Local Search Function
    def check_girth(adj, g):
        n = len(adj)
        for u in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[u] = 0
            queue = [u]
            head = 0
            while head < len(queue):
                v = queue[head]
                head += 1
                if dist[v] >= g // 2:
                    continue
                for w in adj[v]:
                    if dist[w] == -1:
                        dist[w] = dist[v] + 1
                        parent[w] = v
                        queue.append(w)
                    elif w != parent[v] and parent[w] != v:
                        # Found a cycle
                        cycle_len = dist[v] + dist[w] + 1
                        if cycle_len < g:
                            return False
        return True

    def generate_random_k_regular(n, k):
        # Configuration model
        half_edges = []
        for i in range(n):
            for _ in range(k):
                half_edges.append(i)
        
        while True:
            random.shuffle(half_edges)
            adj = [[] for _ in range(n)]
            valid = True
            for i in range(0, len(half_edges), 2):
                u = half_edges[i]
                v = half_edges[i+1]
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

    def find_graph_local_search(k, g, n, time_budget=5.0):
        start = time.time()
        max_restarts = 50
        
        for _ in range(max_restarts):
            if time.time() - start > time_budget:
                break
            
            adj = generate_random_k_regular(n, k)
            
            # Local search
            current_girth = 0
            # Calculate initial girth (approx)
            # We just check if valid
            if check_girth(adj, g):
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return edges
            
            # Hill climbing: try to increase girth
            # We don't have a metric for girth, so we just check validity.
            # If invalid, we need to break short cycles.
            # Strategy: Find a short cycle, swap an edge to break it.
            
            # For simplicity and speed, we just do random restarts if we can't easily find the cycle.
            # A more advanced approach:
            # 1. Find a short cycle.
            # 2. Pick an edge on the cycle.
            # 3. Swap it with a random non-cycle edge.
            
            # Let's implement a simple restart loop with a few swaps
            for _ in range(100):
                if time.time() - start > time_budget:
                    break
                if check_girth(adj, g):
                    edges = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                edges.append((u, v))
                    return edges
                
                # Perform a random valid swap
                # Pick two edges
                all_edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            all_edges.append((u, v))
                
                if len(all_edges) < 2:
                    break
                
                e1, e2 = random.sample(all_edges, 2)
                u1, v1 = e1
                u2, v2 = e2
                
                # Try swap (u1, u2) and (v1, v2)
                # Remove e1, e2
                adj[u1].remove(v1)
                adj[v1].remove(u1)
                adj[u2].remove(v2)
                adj[v2].remove(u2)
                
                # Add new edges
                if u1 != u2 and v1 != v2 and u1 != v2 and v1 != u2:
                    # Check for multi-edges
                    if u2 not in adj[u1] and v2 not in adj[v1]:
                        adj[u1].append(u2)
                        adj[u2].append(u1)
                        adj[v1].append(v2)
                        adj[v2].append(v1)
                    else:
                        # Revert
                        adj[u1].append(v1)
                        adj[v1].append(u1)
                        adj[u2].append(v2)
                        adj[v2].append(u2)
                else:
                    # Revert
                    adj[u1].append(v1)
                    adj[v1].append(u1)
                    adj[u2].append(v2)
                    adj[v2].append(u2)

        return None

    # Main Search Loop
    # Try N from n_min upwards
    # Limit N to n_min + 200 to avoid infinite loops if girth is hard to achieve
    max_n = n_min + 300
    
    for n in range(n_min, max_n):
        if time.time() > deadline:
            break
        
        # Allow less time for smaller n if we are just checking?
        # No, smaller n is harder. Give it full budget.
        # But if we are at n_min, it's very hard.
        # Let's use a fixed budget of 5s per n.
        # If we find it, return.
        
        edges = find_graph_local_search(k, g, n, time_budget=5.0)
        if edges:
            return n, edges

    # Fallback
    return luw_fallback()
