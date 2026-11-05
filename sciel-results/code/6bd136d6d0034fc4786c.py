import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for fallback and cleanup
    
    # Moore bound for k-regular graph of girth g
    if g % 2 == 0:
        m = g // 2
        n_min = 2 * sum((k-1)**i for i in range(m))
    else:
        m = g // 2
        n_min = (k-1)**m + k * sum((k-1)**i for i in range(m))
        
    if (n_min * k) % 2 != 0:
        n_min += 1
        
    max_n = 320
    
    # We will try to find a graph with n close to n_min.
    # If we fail, we use the fallback.
    
    for n in range(n_min, max_n + 1):
        if time.time() > deadline - 10:  # Stop trying new n if close to deadline
            break
            
        # Try multiple random restarts for this n
        # Limit restarts to avoid infinite loops
        max_restarts = 50 
        for r in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            adj = [set() for _ in range(n)]
            deg = [0] * n
            success = True
            
            while True:
                if time.time() > deadline - 2:
                    success = False
                    break
                    
                candidates = [i for i in range(n) if deg[i] < k]
                if not candidates:
                    break # All degrees are k
                    
                u = random.choice(candidates)
                
                # Efficient distance calculation: BFS from u once
                # dist[v] will be the shortest distance from u to v
                # If dist[v] >= g-1, then adding edge (u,v) creates a cycle of length >= g
                # (Actually, if dist is the shortest path in current graph, adding edge creates cycle of length dist+1.
                # We need dist+1 >= g => dist >= g-1.
                # If v is not reachable within g-2, dist is effectively infinity or > g-2.
                # So we check if dist[v] is not found within depth g-2, or if found, dist[v] >= g-1.)
                
                dist = [-1] * n
                dist[u] = 0
                queue = deque([u])
                # BFS up to depth g-2
                while queue:
                    curr = queue.popleft()
                    d_curr = dist[curr]
                    if d_curr >= g - 2:
                        continue
                    for v in adj[curr]:
                        if dist[v] == -1:
                            dist[v] = d_curr + 1
                            if dist[v] < g - 1: # Only need to know if it's < g-1
                                queue.append(v)
                            # If dist[v] >= g-1, we don't need to expand further for this check, 
                            # but we need to know it's "far".
                            # Actually, if we don't visit it, dist remains -1.
                            # We treat -1 as "far enough" (>= g-1) if we stop BFS at g-2.
                            # Wait, if we stop at g-2, any node not visited has dist > g-2, i.e., >= g-1.
                            # So valid_vs are those with dist[v] == -1 OR dist[v] >= g-1.
                            # Since we stop expanding at g-2, any node with dist[v] < g-1 is visited.
                            # Nodes with dist[v] >= g-1 are NOT visited (or visited at g-1? No, we stop expanding at g-2, so g-1 nodes are discovered but not expanded? 
                            # Let's trace:
                            # d=0: u. Expand.
                            # d=1: neighbors. Expand.
                            # ...
                            # d=g-2: nodes at dist g-2. Expand?
                            # If we expand d=g-2, we find nodes at d=g-1.
                            # We should stop expanding at d=g-2.
                            # So nodes at d=g-1 are found and marked, but not expanded.
                            # Nodes at d>=g-1 are NOT found?
                            # If we don't expand d=g-2, we don't find d=g-1.
                            # So d=g-1 nodes remain -1.
                            # So valid_vs are those with dist[v] == -1.
                            # AND those with dist[v] >= g-1 (if we did find them).
                            # But if we stop expanding at g-2, we don't find g-1.
                            # So we just check if dist[v] == -1.
                            pass
                
                # Refine BFS logic:
                # We want v such that shortest path u->v in current graph is >= g-1.
                # If we do BFS from u up to depth g-2:
                # - If v is visited, dist[v] <= g-2. So dist+1 <= g-1. Cycle length < g. INVALID.
                # - If v is NOT visited, dist[v] > g-2 => dist[v] >= g-1. Cycle length >= g. VALID.
                # So we just need to identify which candidates were NOT visited in BFS depth g-2.
                
                valid_vs = []
                for v in candidates:
                    if v == u:
                        continue
                    if v in adj[u]:
                        continue
                    if dist[v] == -1: # Not reached within g-2
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
                # Verify
                for i in range(n):
                    if len(adj[i]) != k:
                        success = False
                        break
                if success:
                    edges = []
                    for i in range(n):
                        for j in adj[i]:
                            if i < j:
                                edges.append((i, j))
                    return n, edges

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
