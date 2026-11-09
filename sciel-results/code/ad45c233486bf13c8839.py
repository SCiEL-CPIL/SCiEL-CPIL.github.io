import time
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # 1. Try Generalized Petersen Graphs G(n, h)
    # Moore bound for N=2n:
    # M(3, g) = 1 + 3 * sum(2^i for i in range(g//2))
    # For g=16, M = 1 + 3 * (2^8 - 1) = 766.
    # So n >= 383.
    
    # We search for n in a reasonable range.
    # If we find a GPG with girth >= g, we return it.
    
    def get_girth_GPG(n, h):
        # n is the number of outer vertices. Total vertices 2n.
        # h is the step for inner vertices.
        if h == 1:
            # G(n, 1) is a ladder graph? No, G(n,1) has triangles if n=3?
            # Inner edges (i, i+1). Outer (i, i+1). Spokes (i, i).
            # Triangle: (i, outer), (i+1, outer), (i+1, inner)?
            # Edges: (i, i+1) outer. (i, i+1) inner. (i, n+i) spoke.
            # Cycle: i(outer) -> i+1(outer) -> i+1(inner) -> i(inner) -> i(outer). Length 4.
            # So girth is 4 for n>2?
            return 4 if n > 2 else 3
            
        # BFS from each vertex
        # Optimization: Due to symmetry, we might not need all starts, but let's do all for safety.
        # We can stop if we find a cycle shorter than g.
        
        V = 2 * n
        # dist[v] = distance from start
        # prev[v] = previous vertex in BFS tree
        # We do BFS for each start node.
        
        # To speed up, we can use a global visited array and reset it?
        # Or just use a list of -1.
        
        min_girth = 2 * n + 1 # Max possible girth in simple graph is V
        
        for start in range(V):
            dist = [-1] * V
            prev = [-1] * V
            dist[start] = 0
            q = deque([start])
            
            while q:
                u = q.popleft()
                du = dist[u]
                
                # Determine neighbors of u
                if u < n:
                    # Outer vertex u
                    # Neighbors: (u-1)%n, (u+1)%n, n+u
                    neighbors = [(u - 1) % n, (u + 1) % n, n + u]
                else:
                    # Inner vertex u, let i = u - n
                    i = u - n
                    # Neighbors: i, n + (i-h)%n, n + (i+h)%n
                    neighbors = [i, n + (i - h) % n, n + (i + h) % n]
                
                for v in neighbors:
                    if dist[v] == -1:
                        dist[v] = du + 1
                        prev[v] = u
                        q.append(v)
                    else:
                        # If v is not the parent of u, we found a cycle
                        if v != prev[u]:
                            # Cycle length is dist[u] + dist[v] + 1
                            # But we need to ensure this is a simple cycle.
                            # In BFS, if we encounter a visited node, the cycle length
                            # is dist[u] + dist[v] + 1 only if the paths are disjoint.
                            # This is a standard heuristic for girth estimation, 
                            # but for exact girth, we must be careful.
                            # However, for small girths, the first back-edge found 
                            # in BFS from all starts usually gives the girth.
                            # To be safe, we can just check if this cycle length < min_girth.
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < min_girth:
                                min_girth = cycle_len
                                # If we found a cycle of length < g, we can break early for this start?
                                # No, we want the MIN girth. If we found a small one, 
                                # we can't break the whole search for this n,h yet, 
                                # but we can break this BFS if we are just checking "is girth >= g?".
                                # We only care if girth >= g.
                                # So if min_girth < g, we can return min_girth immediately?
                                # No, there might be a smaller one. But we only need to know if it's < g.
                                # Actually, if we just want to know if girth >= g, 
                                # we can stop as soon as we find ANY cycle of length < g.
                                # So we can return cycle_len if cycle_len < g.
                                if cycle_len < g:
                                    return cycle_len
                                
            # If we finished BFS for this start, no new cycles found (or they were longer)
        
        return min_girth

    # Search for n
    # Start n from Moore bound / 2
    # M(3, 16) = 766. n_min = 383.
    # We can try a range of n.
    # If we find one, return it.
    
    # To avoid timeout, limit the search.
    # Try n from 383 to 600.
    # For each n, try h from 1 to n-1.
    
    # Optimization: h and n-h give isomorphic graphs.
    # So we only need h up to n//2.
    
    n_start = 383
    n_end = 600
    
    for n in range(n_start, n_end):
        if time.time() > deadline - 5:
            break
        for h in range(1, n // 2 + 1):
            if time.time() > deadline - 5:
                break
            girth_val = get_girth_GPG(n, h)
            if girth_val >= g:
                # Construct the graph
                edges = []
                V = 2 * n
                for i in range(n):
                    # Outer edges
                    edges.append((i, (i + 1) % n))
                    # Spokes
                    edges.append((i, n + i))
                    # Inner edges
                    # (n+i, n+(i+h)%n)
                    # Add only if not already added?
                    # The loop i=0..n-1 adds all inner edges.
                    # (n+i, n+(i+h)%n) is the same as (n+(i+h)%n, n+i).
                    # So we just add them.
                    edges.append((n + i, n + (i + h) % n))
                
                return V, edges

    # 2. Fallback to LUW if no GPG found
    # (Code from seed)
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
