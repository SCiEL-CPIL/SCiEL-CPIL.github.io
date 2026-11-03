import time
import random
import math
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s limit
    
    # Helper to generate random k-regular graph
    def random_regular_graph(n, k):
        if n * k % 2 != 0:
            return None
        # Configuration model
        stubs = []
        for i in range(n):
            stubs.extend([i] * k)
        random.shuffle(stubs)
        
        adj = [set() for _ in range(n)]
        valid = True
        for i in range(0, n * k, 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                valid = False
                break
            if v in adj[u]:
                valid = False
                break
            adj[u].add(v)
            adj[v].add(u)
        
        if not valid:
            return None
        return adj

    def check_girth_at_least(adj, n, g):
        # Returns True if girth >= g, False otherwise
        # BFS from each node, check for cycles shorter than g
        # Max depth to search: (g-1)//2
        max_depth = (g - 1) // 2
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            q = deque([s])
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
                        # Found a cycle
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
        return True

    def swap_edges(adj, n, k):
        # Pick two random edges and swap them
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
        
        if len(edges) < 2:
            return False
        
        idx1 = random.randint(0, len(edges) - 1)
        idx2 = random.randint(0, len(edges) - 1)
        while idx2 == idx1:
            idx2 = random.randint(0, len(edges) - 1)
        
        u1, v1 = edges[idx1]
        u2, v2 = edges[idx2]
        
        options = [
            ((u1, u2), (v1, v2)),
            ((u1, v2), (v1, u2))
        ]
        
        random.shuffle(options)
        
        for e1, e2 in options:
            a1, b1 = e1
            a2, b2 = e2
            if a1 == b1 or a2 == b2:
                continue
            if b1 in adj[a1] or b2 in adj[a2]:
                continue
            if a1 in adj[b1] or a2 in adj[b2]:
                continue
            
            # Perform swap
            adj[u1].remove(v1)
            adj[v1].remove(u1)
            adj[u2].remove(v2)
            adj[v2].remove(u2)
            
            adj[a1].add(b1)
            adj[b1].add(a1)
            adj[a2].add(b2)
            adj[b2].add(a2)
            
            return True
        return False

    # LUW Construction
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
    
    # Moore Bound
    if g % 2 == 0:
        r = g // 2
        moore_bound = 2 * sum((k-1)**i for i in range(r))
    else:
        r = (g - 1) // 2
        moore_bound = k * sum((k-1)**i for i in range(r)) + (k-1)**r
        
    n_start = moore_bound
    if k % 2 == 1:
        if n_start % 2 != 0:
            n_start += 1
    
    # We prioritize smaller n
    # Try up to n_luw, but focus on smaller values
    # Limit the range to avoid wasting time on huge n if small ones fail
    # However, if no small graph exists, we must return LUW.
    
    best_n = n_luw
    best_edges = edges_luw
    
    # Search for smaller n
    # We try n values starting from moore_bound
    # To save time, we limit the number of n values we try and the time per n.
    
    max_n_to_try = min(n_luw, n_start + 20) # Try up to 20 more vertices than Moore bound
    
    for n in range(n_start, max_n_to_try + 1):
        if time.time() > deadline - 10.0:
            break
            
        if k % 2 == 1 and n % 2 != 0:
            continue
            
        # Try to find a graph of size n with girth >= g
        # Use Simulated Annealing
        found = False
        
        # Initialize with a few random starts
        num_restarts = 10
        for _ in range(num_restarts):
            if time.time() > deadline - 10.0:
                break
                
            adj = random_regular_graph(n, k)
            if adj is None:
                continue
                
            if check_girth_at_least(adj, n, g):
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges
                
            # SA Loop
            T = 1.0
            cooling = 0.99
            min_T = 0.01
            steps = 2000 # Increase steps
            
            for i in range(steps):
                if time.time() > deadline - 10.0:
                    break
                    
                if not swap_edges(adj, n, k):
                    continue
                    
                if check_girth_at_least(adj, n, g):
                    edges = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                edges.append((u, v))
                    return n, edges
                T *= cooling
                if T < min_T:
                    break

    # If no smaller graph found, return LUW
    return n_luw, edges_luw
