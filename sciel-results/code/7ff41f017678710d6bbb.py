import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Fallback: LUW construction
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

    n_luw, edges_luw = luw_construct(k, g)
    best_n = n_luw
    best_edges = edges_luw

    # If k and g are small enough, we can try random search.
    # For k=5, g=9, Moore bound is 425.
    # We will try to find a graph with n < best_n.
    
    def check_girth(n, edges, g):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Check degree
        for i in range(n):
            if len(adj[i]) != k:
                return False
        
        d = (g - 1) // 2
        
        for v in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[v] = 0
            queue = [(v, 0)]
            
            while queue:
                u, depth = queue.pop(0)
                if depth == d:
                    continue
                for w in adj[u]:
                    if dist[w] == -1:
                        dist[w] = depth + 1
                        parent[w] = u
                        queue.append((w, depth + 1))
                    else:
                        if parent[u] == w or parent[w] == u:
                            continue
                        cycle_len = dist[u] + dist[w] + 1
                        if cycle_len < g:
                            return False
                            
        return True

    def generate_random_k_regular(n, k):
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        
        random.shuffle(stubs)
        
        edges = []
        for i in range(0, n * k, 2):
            u = stubs[i]
            v = stubs[i + 1]
            if u == v:
                return None
            edges.append((u, v))
            
        edge_set = set()
        for u, v in edges:
            if u > v:
                u, v = v, u
            if (u, v) in edge_set:
                return None
            edge_set.add((u, v))
            
        return edges

    # Search strategy
    # Try to find a graph with n < best_n.
    # We will try specific target n values.
    # Known best is 1116.
    # We will try to find one with n=1116, then 1100, 1000, etc.
    
    # To be safe, we only do random search if best_n is large enough.
    if best_n > 500:
        # We will try a few target n values.
        # We will use a deadline.
        
        # Target n values to try
        # We'll pick a random n in [500, best_n - 1]
        # But we know 1116 works. So let's try 1116 first.
        
        targets = [1116, 1100, 1000, 900, 800, 700, 600]
        targets = [t for t in targets if t < best_n and t >= 425]
        
        random.shuffle(targets)
        
        for target_n in targets:
            if time.time() > deadline - 10.0:
                break
            
            # Try to find a valid graph of size target_n
            found = False
            # Limit time per target
            target_deadline = min(deadline - 10.0, time.time() + 10.0)
            
            while time.time() < target_deadline:
                edges = generate_random_k_regular(target_n, k)
                if edges is not None:
                    if check_girth(target_n, edges, g):
                        if target_n < best_n:
                            best_n = target_n
                            best_edges = edges
                        found = True
                        break
                            
    return best_n, best_edges
