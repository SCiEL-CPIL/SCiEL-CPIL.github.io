import time
import math
import random

def construct(k, g):
    t0 = time.time()
    deadline = t0 + 110  # Safety margin

    # Moore Bound
    if g % 2 == 0:
        h = g // 2
    else:
        h = (g - 1) // 2
    moore_bound = 1 + k * sum((k - 1) ** i for i in range(h))

    # Search for Circulant Graph C_n(1, d)
    # Vertices 0..n-1
    # Edges (i, i+1) and (i, i+d)
    
    # Helper to check girth of C_n(1, d)
    def check_circulant_girth(n, d, required_girth):
        # Build adjacency list
        # Since it's circulant, we can just use modulo arithmetic in BFS
        # dist[v] = distance from 0
        # parent[v] = parent in BFS tree
        # We want min cycle length containing 0.
        # In BFS tree, any non-tree edge (u, v) creates a cycle of length dist[u] + dist[v] + 1.
        
        dist = [-1] * n
        parent = [-1] * n
        dist[0] = 0
        queue = [0]
        head = 0
        
        min_cycle = float('inf')
        
        while head < len(queue):
            u = queue[head]
            head += 1
            
            # Neighbors of u in C_n(1, d)
            # 1. (u+1) % n
            # 2. (u-1) % n
            # 3. (u+d) % n
            # 4. (u-d) % n
            
            neighbors = [(u + 1) % n, (u - 1) % n, (u + d) % n, (u - d) % n]
            
            for v in neighbors:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    # If v is already visited
                    # If v is the parent of u, skip
                    if parent[u] == v:
                        continue
                    # Otherwise, (u, v) is a non-tree edge (or back edge)
                    # Cycle length = dist[u] + dist[v] + 1
                    # This cycle contains 0?
                    # The path 0->...->u, edge u-v, path v->...->0.
                    # Yes, it contains 0.
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_cycle:
                        min_cycle = cycle_len
                        if min_cycle < required_girth:
                            return min_cycle # Early exit if too small
        
        return min_cycle

    # Iterate n
    # Start from moore_bound
    # Step size 1? Or 2?
    # For k=3, n must be even?
    # C_n(1,d) is 3-regular.
    # Sum of degrees = 3n. Must be even. So n can be any integer?
    # 3n is even => n is even.
    # So n must be even.
    
    n = moore_bound
    if n % 2 != 0:
        n += 1
        
    # Upper bound for search
    # Let's try up to moore_bound + 1000
    max_n = moore_bound + 1000
    
    while n <= max_n and time.time() < deadline:
        # d must be such that 1, d, n-1, n-d are distinct
        # d != 1, d != n-1, d != n/2
        # Also d != 0
        # d ranges from 2 to n//2 - 1 (if n/2 is integer, skip it)
        
        for d in range(2, n // 2):
            if 2 * d == n:
                continue
            
            # Quick check: if d is small, girth is small?
            # If d=2, C_n(1,2) has triangles?
            # 0-1-2-0 is a triangle if 2-0 is an edge?
            # Edges are (i, i+1) and (i, i+2).
            # 0-1 (step 1), 1-2 (step 1), 2-0 (step -2, which is edge (0,2)).
            # So 0-1-2-0 is a triangle. Girth 3.
            # So d=2 is bad.
            
            # We can skip small d?
            # Girth of C_n(1,d) is at least 2*min(d, n-d) ? No.
            # Actually, if d is small, we get short cycles.
            # Let's just run the check.
            
            girth = check_circulant_girth(n, d, g)
            
            if girth >= g:
                # Found a valid graph!
                unique_edges = set()
                for i in range(n):
                    v1 = (i + 1) % n
                    v2 = (i + d) % n
                    e1 = (min(i, v1), max(i, v1))
                    e2 = (min(i, v2), max(i, v2))
                    unique_edges.add(e1)
                    unique_edges.add(e2)
                
                return n, list(unique_edges)
        
        n += 2

    # Fallback: L-U-W Construction
    # (Copy from seed)
    t0 = time.time()
    q = k
    d_luw = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d_luw + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d_luw - 1)
            for c in range(2, d_luw + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d_luw)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        if time.time() - t0 > 140:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
