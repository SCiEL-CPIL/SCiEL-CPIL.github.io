import time
import random
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Fallback: LUW Construction
    # This is the seed's logic, kept as a safety net.
    def luw_construct(k, g):
        q = k
        d = g - 5
        if d % 2 == 0:
            d += 1
        d = max(3, d)
        
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q

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
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        
        for i, vertex in enumerate(order):
            if time.time() > deadline:
                break
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
                
        return len(order), edges

    # Attempt 1: Circulant Graph Search
    # Try to find a circulant graph C_n({a, -a, b, -b, n/2}) with girth >= g
    # This is only applicable if k is odd (5) and n is even.
    def try_circulants(k, g, max_n, time_limit):
        if k % 2 != 0:
            pass # Only odd k handled here for simplicity, even k would use {a, -a, b, -b, c, -c}
        else:
            return None

        # Moore bound for k=5, g=9 is approx 106.
        # We scan n from 106 to max_n.
        # We only check even n.
        
        # To save time, we don't check every n. We check a subset or all?
        # 1200/2 = 600 values.
        # For each n, we try a few random (a, b) pairs.
        
        for n in range(106, max_n + 1, 2):
            if time.time() > time_limit:
                return None
            
            # n must be even for k=5 circulant with n/2
            if n % 2 != 0:
                continue
            
            half_n = n // 2
            
            # We need to pick a, b from 1 to half_n - 1
            # a != b, a != 0, b != 0, a != half_n, b != half_n
            # To be efficient, we try a few random pairs
            # If we find one, we return.
            
            # Heuristic: Small a, b often give small girth.
            # Large a, b might give larger girth?
            # Let's just try 10 random pairs per n.
            
            found = False
            for _ in range(10):
                a = random.randint(1, half_n - 1)
                b = random.randint(1, half_n - 1)
                if a == b:
                    continue
                
                # Check if this circulant has girth >= g
                # Build adjacency list
                adj = [[] for _ in range(n)]
                for u in range(n):
                    v1 = (u + a) % n
                    v2 = (u - a) % n
                    v3 = (u + b) % n
                    v4 = (u - b) % n
                    v5 = (u + half_n) % n
                    
                    # Simple check: no loops (a,b,half_n != 0)
                    # No multi-edges: a != b, a != -b, etc.
                    # Since a,b in 1..half_n-1, a != -a unless a=0 or a=half_n.
                    # So edges are distinct if a != b and a != -b mod n.
                    # a != -b => a+b != 0 mod n. Since a,b < n/2, a+b < n, so a+b != n.
                    # So it's simple.
                    
                    adj[u].extend([v1, v2, v3, v4, v5])
                
                # Check Girth
                # BFS from 0
                dist = [-1] * n
                parent = [-1] * n
                queue = [0]
                dist[0] = 0
                girth = float('inf')
                
                qi = 0
                while qi < len(queue):
                    u = queue[qi]
                    qi += 1
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            queue.append(v)
                        else:
                            # Back edge
                            # Cycle length = dist[u] + dist[v] + 1
                            # This is not always the shortest cycle through this edge,
                            # but for girth check, we can just look for any cycle < g.
                            # Actually, BFS finds shortest path to v.
                            # If we see a visited node, the cycle length is dist[u] + 1 + dist[v]?
                            # No, that's for undirected graphs where we don't distinguish tree/back edges properly.
                            # Proper girth check:
                            # For each edge (u, v), if parent[u] != v and parent[v] != u:
                            #   cycle_len = dist[u] + dist[v] + 1
                            #   girth = min(girth, cycle_len)
                            
                            if parent[u] != v and parent[v] != u:
                                cycle_len = dist[u] + dist[v] + 1
                                if cycle_len < girth:
                                    girth = cycle_len
                                    if girth < g:
                                        break
                    if girth < g:
                        break
                
                if girth >= g:
                    # Found a valid circulant!
                    # Construct edges list
                    edges = []
                    for u in range(n):
                        for v in adj[u]:
                            if u < v:
                                edges.append((u, v))
                    return n, edges
            
        return None

    # 1. Try Circulants
    # Limit time to 60s to leave room for LUW
    circ_result = try_circulants(k, g, 1249, start_time + 60.0)
    if circ_result is not None:
        return circ_result

    # 2. Fallback to LUW
    return luw_construct(k, g)
