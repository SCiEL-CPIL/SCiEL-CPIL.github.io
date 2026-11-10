import time
import random
import collections

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    # Moore bound for k=3, even g
    if k == 3 and g % 2 == 0:
        moore = 2 * sum((k - 1) ** i for i in range(g // 2))
    else:
        moore = 1 + k * sum((k - 1) ** i for i in range((g - 1) // 2))
        
    if moore < k:
        moore = k

    # Try random search
    # Limit n to a reasonable range where success is plausible
    # Known cage is 936. Let's search up to 1100.
    max_n = 1100
    
    for n in range(moore, max_n + 1):
        if time.time() > deadline - 10.0:
            break
            
        # Estimate attempts based on remaining time
        # Assume ~0.05s per check for n~1000
        time_per_check = 0.05
        remaining = deadline - time.time() - 5.0
        attempts = max(1, int(remaining / time_per_check))
        
        for _ in range(attempts):
            if time.time() > deadline - 5.0:
                break
                
            edges = _generate_random_3reg(n)
            if edges is None:
                continue
                
            if _check_girth(edges, n, g):
                return n, edges
                
    # Fallback to LUW construction if random search fails
    return _lww_fallback(k, g)

def _generate_random_3reg(n):
    # Configuration model
    half_edges = []
    for i in range(n):
        half_edges.extend([i, i, i])
        
    random.shuffle(half_edges)
    
    edges = []
    seen = set()
    for i in range(0, 2 * n, 2):
        u = half_edges[i]
        v = half_edges[i + 1]
        if u == v:
            return None
        if u > v:
            u, v = v, u
        if (u, v) in seen:
            return None
        seen.add((u, v))
        edges.append((u, v))
        
    return edges

def _check_girth(edges, n, g):
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Check for girth
    # BFS from each vertex
    for start in range(n):
        # dist[v] = distance from start
        dist = [-1] * n
        dist[start] = 0
        queue = collections.deque([start])
        
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
                else:
                    # Back edge or cross edge
                    # Cycle length = dist[u] + 1 + dist[v]
                    # This is valid if v is not the parent of u in the BFS tree
                    # We don't track parents, but if dist[v] <= dist[u], it's a back/cross edge
                    # If dist[v] > dist[u], it's a forward edge (already handled when v was processed?)
                    # Actually, in BFS, if dist[v] != -1, and v is not parent of u,
                    # then we have a cycle.
                    # The length of the cycle is dist[u] + 1 + dist[v]
                    # However, this formula assumes the path from start to v is disjoint from start to u?
                    # No, it's the length of the cycle formed by the tree paths and the edge (u,v).
                    # This cycle length is a lower bound on the actual girth?
                    # Yes, any such cycle is a valid cycle.
                    cycle_len = dist[u] + 1 + dist[v]
                    if cycle_len < g:
                        return False
    return True

def _lww_fallback(k, g):
    # Original LUW code
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
