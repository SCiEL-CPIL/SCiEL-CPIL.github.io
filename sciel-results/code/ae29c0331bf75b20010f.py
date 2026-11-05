import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110

    # Modular arithmetic construction on Z_n x Z_m
    def modular_construct(k, g, n, m, coeffs):
        """
        Construct a k-regular graph on Z_n x Z_m with adjacency defined by:
        (i,j) ~ (i+c_i, j+d_i) for each coefficient pair (c_i, d_i)
        and also (i,j) ~ (i-c_i, j-d_i) for non-zero coefficients.
        We need exactly k neighbors, so we choose coeffs carefully.
        """
        total_vertices = n * m
        
        # Build adjacency
        adj = [set() for _ in range(total_vertices)]
        
        def idx(i, j):
            return i * m + j
        
        for i in range(n):
            for j in range(m):
                v = idx(i, j)
                for c, d in coeffs:
                    # Add neighbor in positive direction
                    ni = (i + c) % n
                    nj = (j + d) % m
                    u = idx(ni, nj)
                    adj[v].add(u)
                    # Add neighbor in negative direction (if different)
                    ni2 = (i - c) % n
                    nj2 = (j - d) % m
                    u2 = idx(ni2, nj2)
                    if u2 != v:  # Avoid self-loops
                        adj[v].add(u2)
        
        # Check if all vertices have degree exactly k
        for v in range(total_vertices):
            if len(adj[v]) != k:
                return None
        
        # Check for girth >= g using BFS from each vertex
        # But this is expensive, so we use a smarter approach:
        # Check for cycles of length < g
        
        def has_short_cycle():
            # For each vertex, do BFS up to depth (g-1)//2
            max_depth = (g - 1) // 2
            
            for start in range(total_vertices):
                # BFS from start
                dist = [-1] * total_vertices
                parent = [-1] * total_vertices
                dist[start] = 0
                queue = deque([start])
                
                while queue:
                    u = queue.popleft()
                    if dist[u] >= max_depth:
                        continue
                    
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            queue.append(v)
                        else:
                            # Found a cycle
                            # Cycle length is dist[u] + dist[v] + 1
                            # But we need to be careful about the actual cycle
                            # In BFS, if we find an edge to an already visited node,
                            # the cycle length is at most dist[u] + dist[v] + 1
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g:
                                return True
                
                # Also check if we can return to start
                # This is handled by the above if we consider edges back to start
        
            return False
        
        if has_short_cycle():
            return None
        
        # Build edge list
        edges = []
        for u in range(total_vertices):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
        
        return total_vertices, edges

    # Fallback: LUW graph (current implementation)
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

    # Try to find a smaller graph using modular construction on Z_n x Z_m
    if k == 5 and g == 9:
        # For k=5, we need 5 neighbors.
        # Use coeffs that give exactly 5 distinct neighbors.
        # Strategy: use one "reflection" coefficient (c,d) where (c,d) = (-c,-d) mod (n,m),
        # i.e., 2c ≡ 0 mod n and 2d ≡ 0 mod m, and two other coefficients.
        # But this is complex. Let's try a simpler approach:
        # Use a circulant-like structure on Z_n with carefully chosen generators.
        
        def check_girth_cyclic_fast(n, a, b):
            """Check if Z_n with generators {a, -a, b, -b, n/2} has girth >= 9"""
            c = n // 2
            S = [a, n - a, b, n - b, c]
            
            # Check for distinct generators
            if len(set(S)) != 5:
                return False
            if 0 in S:
                return False
            
            # BFS from 0 to depth 4
            dist = [-1] * n
            dist[0] = 0
            queue = deque([0])
            
            while queue:
                u = queue.popleft()
                if dist[u] >= 4:
                    continue
                for s in S:
                    v = (u + s) % n
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
                    else:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < 9:
                            return False
            
            return True
        
        # Search for valid (n, a, b)
        # n must be even (for the n/2 generator)
        # Start from smaller n and increase
        for n in range(428, 1250, 2):
            if time.time() > deadline - 5:
                break
            
            # Try random pairs (a, b)
            max_tries = 30
            for _ in range(max_tries):
                a = random.randint(1, n // 2 - 1)
                b = random.randint(1, n // 2 - 1)
                if a == b or a == n - b:
                    continue
                if (a + b) % n == 0 or (a - b) % n == 0:
                    continue
                
                if check_girth_cyclic_fast(n, a, b):
                    # Found a valid graph!
                    c = n // 2
                    S = [a, n - a, b, n - b, c]
                    edges = []
                    seen = set()
                    for u in range(n):
                        for s in S:
                            v = (u + s) % n
                            if u < v:
                                key = (u, v)
                                if key not in seen:
                                    edges.append(key)
                                    seen.add(key)
                    return n, edges
                    
    # Fallback to LUW
    return luw_construct(k, g)
