import numpy as np
import random
import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    time_limit = 115  # Leave buffer
    
    # Moore bound
    def moore_bound(k, g):
        if k == 2:
            return g
        if g % 2 == 1:
            terms = (g - 1) // 2
            if k == 1:
                return g
            if k == 2:
                return 1 + 2 * terms
            return 1 + k * (( (k-1)**terms - 1 ) // (k - 2))
        else:
            terms = g // 2 - 1
            if k == 2:
                return 2 * terms
            return 2 * (( (k-1)**terms - 1 ) // (k - 2))
    
    n_min = moore_bound(k, g)
    
    # Try different n values starting from a reasonable estimate
    # The actual cage size is often between Moore bound and Moore bound * 2-3
    
    def generate_random_regular_graph(n, k):
        """Generate a random k-regular simple graph using configuration model."""
        if n * k % 2 != 0 or k >= n:
            return None
        
        # Use configuration model with rejection
        for _ in range(100):
            stubs = []
            for i in range(n):
                stubs.extend([i] * k)
            random.shuffle(stubs)
            
            adj = [set() for _ in range(n)]
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i+1]
                if u == v:
                    valid = False
                    break
                if v in adj[u]:
                    valid = False
                    break
                adj[u].add(v)
                adj[v].add(u)
            
            if valid and all(len(adj[i]) == k for i in range(n)):
                return [list(s) for s in adj]
        
        return None
    
    def find_short_cycle(adj, g):
        """Find a cycle of length < g using BFS from each vertex."""
        n = len(adj)
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
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            # Reconstruct cycle
                            cycle = [u]
                            curr = u
                            while curr != v:
                                curr = parent[curr]
                                cycle.append(curr)
                            cycle.append(v)
                            return cycle
        return None
    
    def try_swap(adj, u, v, x, y):
        """Try to swap edges (u,v) and (x,y) with (u,x) and (v,y) or (u,y) and (v,x)."""
        n = len(adj)
        
        # Check validity conditions
        # 1. No self-loops
        # 2. No multi-edges
        # 3. u != x, u != y, v != x, v != y (edges must be disjoint)
        
        if u == x or u == y or v == x or v == y:
            return None
        
        # Try swap 1: (u,v) -> (u,x), (x,y) -> (v,y)
        # Remove (u,v) and (x,y), add (u,x) and (v,y)
        if x not in adj[u] and y not in adj[v]:
            new_adj = [list(s) for s in adj]
            new_adj[u].remove(v)
            new_adj[v].remove(u)
            new_adj[x].remove(y)
            new_adj[y].remove(x)
            new_adj[u].append(x)
            new_adj[x].append(u)
            new_adj[v].append(y)
            new_adj[y].append(v)
            return new_adj
        
        # Try swap 2: (u,v) -> (u,y), (x,y) -> (v,x)
        if y not in adj[u] and x not in adj[v]:
            new_adj = [list(s) for s in adj]
            new_adj[u].remove(v)
            new_adj[v].remove(u)
            new_adj[x].remove(y)
            new_adj[y].remove(x)
            new_adj[u].append(y)
            new_adj[y].append(u)
            new_adj[v].append(x)
            new_adj[x].append(v)
            return new_adj
        
        return None
    
    def local_search(n, k, g, time_limit):
        """Run local search to find a k-regular graph with girth >= g."""
        adj = None
        for _ in range(50):
            adj = generate_random_regular_graph(n, k)
            if adj:
                break
        
        if not adj:
            return None
        
        # Precompute adjacency sets for fast lookup
        adj_sets = [set(neigh) for neigh in adj]
        
        # Get all edges
        def get_edges(adj):
            edges = set()
            for i in range(len(adj)):
                for j in adj[i]:
                    if i < j:
                        edges.add((i, j))
            return edges
        
        # Main loop
        for step in range(100000):
            if time.time() - start_time > time_limit:
                break
            
            # Find a short cycle
            cycle = find_short_cycle(adj, g)
            if cycle is None:
                return adj
            
            # Pick a random edge from the cycle
            edge_idx = random.randint(0, len(cycle) - 1)
            u = cycle[edge_idx]
            v = cycle[(edge_idx + 1) % len(cycle)]
            
            # Try to swap this edge with a random other edge
            success = False
            for _ in range(20):
                x = random.randint(0, n - 1)
                y = random.randint(0, n - 1)
                if x == y or x == u or x == v or y == u or y == v:
                    continue
                if (x, y) not in adj_sets[x] and (y, x) not in adj_sets[y]:
                    continue
                if x > y:
                    x, y = y, x
                if u > v:
                    u, v = v, u
                if (x, y) == (u, v):
                    continue
                
                new_adj = try_swap(adj, u, v, x, y)
                if new_adj:
                    # Check if this improves the girth (heuristic: check if it breaks the cycle)
                    # For simplicity, accept the swap if it's valid
                    adj = new_adj
                    adj_sets = [set(neigh) for neigh in adj]
                    success = True
                    break
            
            if not success:
                # Try more random swaps
                continue
        
        return adj
    
    # Try different n values
    best_result = None
    for n in range(n_min, n_min + 100):
        if n * k % 2 != 0:
            continue
        if time.time() - start_time > time_limit - 10:
            break
        
        result = local_search(n, k, g, time_limit)
        if result:
            # Verify the result
            n_vert = len(result)
            if all(len(result[i]) == k for i in range(n_vert)):
                # Check girth
                if find_short_cycle(result, g) is None:
                    edges = []
                    for i in range(n_vert):
                        for j in result[i]:
                            if i < j:
                                edges.append((i, j))
                    if best_result is None or n_vert < best_result[0]:
                        best_result = (n_vert, edges)
    
    # If local search failed, fall back to LUW
    if best_result is None:
        # LUW construction
        q = k
        d = max(3, (g - 1) // 2)
        
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
        
        best_result = (len(order), edges)
    
    return best_result
