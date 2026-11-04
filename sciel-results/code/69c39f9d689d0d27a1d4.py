import time
import math
import random

_start_time = None

def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def _deadline_exceeded(limit=140):
    return _elapsed() > limit

def construct_lw(k, g):
    # Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k
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

def construct_abs_heisenberg(k, g):
    # Try Heisenberg group H_q construction for odd k
    # Specifically for k=5, use even q
    if k != 5:
        return None
    
    # Estimate q
    # n = q^3
    # Moore bound for k=5, g=9 is 426
    # Try q such that q^3 is reasonable
    # Start from small even q
    candidates = [10, 12, 14, 16, 18, 20]
    
    for q in candidates:
        if _deadline_exceeded():
            return None
            
        n = q**3
        # Generators for k=5 on H_q (even q)
        # g1 = (1, 0, 0)
        # g2 = (q-1, 0, 0)
        # g3 = (0, 1, 0)
        # g4 = (0, q-1, 0)
        # g5 = (0, 0, q//2)
        
        gens = [
            (1, 0, 0),
            (q-1, 0, 0),
            (0, 1, 0),
            (0, q-1, 0),
            (0, 0, q//2)
        ]
        
        # Build adjacency list
        adj = [[] for _ in range(n)]
        
        def idx(x, y, z):
            return (x * q + y) * q + z
            
        for x in range(q):
            for y in range(q):
                for z in range(q):
                    u = idx(x, y, z)
                    for gx, gy, gz in gens:
                        vx = (x + gx) % q
                        vy = (y + gy) % q
                        vz = (z + gz + x * gy) % q
                        v = idx(vx, vy, vz)
                        if u < v:
                            adj[u].append(v)
                            adj[v].append(u)
                            
        # Verify girth
        # Check if any cycle < g
        # For each vertex u, check shortest path between neighbors
        # This is O(n * k^2 * n * k) which is heavy for large n, but n is small here
        
        def has_short_cycle(adj, n, g):
            # BFS from each node is too slow for n=1000?
            # 1000 nodes.
            # For each node u (1000)
            #   For each pair of neighbors (10 pairs)
            #     BFS to find shortest path (1000 nodes)
            # Total ~ 10^7 ops. OK.
            
            # Optimization: Early exit
            for u in range(n):
                neighbors = adj[u]
                # If degree < 2, no cycle through u?
                # k=5, so always 5 neighbors.
                for i in range(len(neighbors)):
                    for j in range(i+1, len(neighbors)):
                        v1 = neighbors[i]
                        v2 = neighbors[j]
                        if v1 == v2: continue
                        
                        # BFS from v1 to v2, excluding u
                        # Use deque for BFS
                        from collections import deque
                        visited = {v1: 0}
                        queue = deque([v1])
                        found_dist = -1
                        
                        while queue:
                            curr = queue.popleft()
                            dist = visited[curr]
                            if dist >= g - 2: 
                                # If we haven't found v2 by dist g-2, 
                                # any path found later will be >= g-2, 
                                # so cycle length >= g.
                                # Wait, we want to know if dist < g-2.
                                # If we reach depth g-2 and haven't found v2, 
                                # then shortest path is >= g-2? 
                                # No, BFS finds shortest.
                                # If we exhaust the BFS up to depth g-3 and don't find v2, 
                                # then shortest path >= g-2.
                                # So we can stop BFS at depth g-3.
                                break
                            
                            for nxt in adj[curr]:
                                if nxt == u:
                                    continue
                                if nxt not in visited:
                                    visited[nxt] = dist + 1
                                    if nxt == v2:
                                        found_dist = dist + 1
                                        queue.clear()
                                        break
                                    if dist + 1 < g - 2:
                                        queue.append(nxt)
                                
                        if found_dist != -1:
                            if found_dist + 2 < g:
                                return True # Found short cycle
                        else:
                            # If we broke out of BFS loop due to depth limit
                            # and didn't find v2, then dist >= g-2
                            # So cycle length >= g.
                            pass
                            
            return False
            
        if not has_short_cycle(adj, n, g):
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            return n, edges
            
    return None

def construct(k, g):
    global _start_time
    _start_time = time.time()
    
    # Try Heisenberg for k=5
    if k == 5:
        result = construct_abs_heisenberg(k, g)
        if result is not None:
            return result
            
    # Fallback to LW
    return construct_lw(k, g)
