import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
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

    # Try local search for smaller N
    # Moore bound for k=3, g=16 is 510.
    # Known best is 936.
    # We try N from 512 to 1200.
    
    best_n = None
    best_edges = None
    best_girth = 0
    
    # Function to compute girth
    def compute_girth(n, adj):
        min_girth = float('inf')
        for start in range(n):
            # BFS to find shortest cycle starting from start
            # dist[u] = distance from start
            # parent[u] = parent in BFS tree
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            for u in queue:
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        # Found a cycle
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
        return min_girth if min_girth != float('inf') else 0

    # Function to generate random 3-regular graph
    def generate_random_3reg(n):
        # Use configuration model
        # Ensure n is even
        if n % 2 != 0:
            n += 1
        half = [i for i in range(n) for _ in range(3)]
        random.shuffle(half)
        adj = [set() for _ in range(n)]
        edges = []
        for i in range(0, n * 3, 2):
            u = half[i]
            v = half[i+1]
            if u == v:
                # Retry or handle loop
                # Simple retry
                while u == v:
                    random.shuffle(half)
                    u = half[i]
                    v = half[i+1]
            if v in adj[u]:
                # Multi-edge, retry
                while v in adj[u]:
                    random.shuffle(half)
                    u = half[i]
                    v = half[i+1]
            adj[u].add(v)
            adj[v].add(u)
            edges.append((u, v))
        return n, adj, edges

    # Local search
    def local_search(n, adj, edges, target_girth):
        current_girth = compute_girth(n, adj)
        if current_girth >= target_girth:
            return n, adj, edges, current_girth
        
        temp = 1.0
        cooling = 0.999
        min_temp = 0.01
        
        while temp > min_temp:
            if time.time() > deadline:
                break
            
            # Find a short cycle
            # We can just pick a random edge and see if it's in a short cycle?
            # Better: Find shortest cycle
            # For efficiency, we might just try random switches and check girth
            # But checking girth is O(N^2).
            
            # Pick two random edges
            # Ensure they are not adjacent
            e1_idx = random.randint(0, len(edges)-1)
            e2_idx = random.randint(0, len(edges)-1)
            if e1_idx == e2_idx:
                continue
            
            u1, v1 = edges[e1_idx]
            u2, v2 = edges[e2_idx]
            
            # Check if edges share vertices
            if len({u1, v1, u2, v2}) < 4:
                continue
            
            # Try switch: (u1, u2) and (v1, v2)
            if u2 in adj[u1] or v2 in adj[v1]:
                continue
            
            # Perform switch
            adj[u1].remove(v1)
            adj[v1].remove(u1)
            adj[u2].remove(v2)
            adj[v2].remove(u2)
            
            adj[u1].add(u2)
            adj[u2].add(u1)
            adj[v1].add(v2)
            adj[v2].add(v1)
            
            edges[e1_idx] = (u1, u2)
            edges[e2_idx] = (v1, v2)
            
            new_girth = compute_girth(n, adj)
            
            if new_girth > current_girth:
                current_girth = new_girth
                if current_girth >= target_girth:
                    return n, adj, edges, current_girth
            elif new_girth == current_girth:
                # Accept with probability 1.0 to explore
                current_girth = new_girth
            else:
                # Reject: revert
                adj[u1].remove(u2)
                adj[u2].remove(u1)
                adj[v1].remove(v2)
                adj[v2].remove(v1)
                
                adj[u1].add(v1)
                adj[v1].add(u1)
                adj[u2].add(v2)
                adj[v2].add(u2)
                
                edges[e1_idx] = (u1, v1)
                edges[e2_idx] = (u2, v2)
            
            temp *= cooling
        
        return n, adj, edges, current_girth

    # Main loop
    for n_attempt in range(512, 1200, 16):
        if time.time() > deadline:
            break
        
        # Try a few restarts for this n
        for restart in range(5):
            if time.time() > deadline:
                break
            
            n, adj, edges = generate_random_3reg(n_attempt)
            n, adj, edges, girth = local_search(n, adj, edges, g)
            
            if girth >= g:
                return n, edges
            
            if best_n is None or n < best_n:
                best_n = n
                best_edges = edges
                best_girth = girth
    
    # If no valid graph found, return LUW
    return luw_construct()
# EVOLVE-BLOCK-END
