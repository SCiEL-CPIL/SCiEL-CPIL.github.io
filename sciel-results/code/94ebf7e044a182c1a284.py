import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Moore bound for k-regular graph of girth g
    # For even g = 2m: n >= 2 * sum_{i=0}^{m-1} (k-1)^i
    # For odd g = 2m+1: n >= (k-1)^m + k * sum_{i=0}^{m-1} (k-1)^i
    if g % 2 == 0:
        m = g // 2
        n_min = 2 * sum((k-1)**i for i in range(m))
    else:
        m = g // 2
        n_min = (k-1)**m + k * sum((k-1)**i for i in range(m))
        
    # Ensure n_min is even for bipartite-like constraints? 
    # Actually, for k-regular graphs, n*k must be even.
    if (n_min * k) % 2 != 0:
        n_min += 1
        
    # We know best known is 320. We search up to 320.
    max_n = 320
    
    for n in range(n_min, max_n + 1):
        if time.time() > deadline:
            break
            
        # Try multiple random restarts for this n
        max_restarts = 1000
        for r in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Initialize adjacency list
            adj = [set() for _ in range(n)]
            deg = [0] * n
            success = True
            
            # Order of vertices to process
            # We'll pick vertices with degree < k
            # To be efficient, we can just iterate through vertices
            # and try to add edges.
            
            # A better strategy: 
            # While there are vertices with degree < k:
            #   Pick a random vertex u with deg[u] < k
            #   Find a valid v with deg[v] < k and dist(u,v) >= g-1
            #   Add edge (u,v)
            
            while True:
                if time.time() > deadline:
                    success = False
                    break
                    
                # Find vertices with degree < k
                candidates = [i for i in range(n) if deg[i] < k]
                if not candidates:
                    break # All degrees are k
                    
                # Pick a random candidate u
                u = random.choice(candidates)
                
                # Find valid v
                valid_vs = []
                for v in candidates:
                    if v == u:
                        continue
                    if v in adj[u]:
                        continue
                    # Check distance
                    # BFS from u to v in current graph
                    # Stop if depth >= g-1
                    # If v is not reached within depth g-2, dist >= g-1
                    dist = _bfs_distance(adj, u, v, g-2)
                    if dist >= g-1: # Actually, if not found within g-2, dist is > g-2, so >= g-1
                        valid_vs.append(v)
                        
                if not valid_vs:
                    success = False
                    break
                else:
                    v = random.choice(valid_vs)
                    adj[u].add(v)
                    adj[v].add(u)
                    deg[u] += 1
                    deg[v] += 1
                    
            if success:
                # Verify the graph is valid
                # Check degrees
                for i in range(n):
                    if len(adj[i]) != k:
                        success = False
                        break
                if success:
                    # Check girth exactly
                    # We constructed it to have no cycles < g, but let's be sure
                    # The BFS check during construction ensures that every added edge
                    # creates a cycle of length >= g.
                    # Since the graph is built by adding edges, any cycle must contain
                    # the last edge added to it.
                    # When that edge (u,v) was added, dist(u,v) in the previous graph was >= g-1.
                    # So the cycle length is dist+1 >= g.
                    # Thus, the girth is >= g.
                    # We just need to output the edges.
                    edges = []
                    for i in range(n):
                        for j in adj[i]:
                            if i < j:
                                edges.append((i, j))
                    return n, edges
        # If all restarts failed for this n, try next n

    # If no graph found, return the LUW graph as fallback (though it should be found)
    # Or raise an error? The task says "return the best result found".
    # If we run out of time, we should return the best valid graph found so far.
    # But the loop structure above returns immediately upon finding a valid one.
    # If we break out of the loop without returning, we need a fallback.
    
    # Fallback: LUW graph
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

def _bfs_distance(adj, start, target, max_depth):
    """
    Returns the shortest path distance from start to target in the graph defined by adj.
    Returns max_depth + 1 if target is not reachable within max_depth.
    """
    if start == target:
        return 0
    visited = {start: 0}
    queue = deque([start])
    while queue:
        u = queue.popleft()
        d = visited[u]
        if d >= max_depth:
            continue
        for v in adj[u]:
            if v not in visited:
                visited[v] = d + 1
                if v == target:
                    return d + 1
                queue.append(v)
    return max_depth + 1
