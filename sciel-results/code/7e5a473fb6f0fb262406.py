import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Moore bound: 1 + k * sum_{i=0}^{floor((g-2)/2)-1} (k-1)^i
    # For even g=2r, bound is 1 + k * ( (k-1)^r - 1 ) / (k-2) ... wait
    # Standard Moore bound for girth g:
    # If g = 2r: n >= 1 + k * sum_{i=0}^{r-1} (k-1)^i
    # If g = 2r+1: n >= k * sum_{i=0}^{r} (k-1)^i
    if (g - 2) % 2 == 0:
        r = (g - 2) // 2
        # 1 + k * (1 + (k-1) + ... + (k-1)^(r-1))
        # Sum is ( (k-1)^r - 1 ) / ( (k-1) - 1 ) = ( (k-1)^r - 1 ) / (k-2)
        if k > 2:
            moore = 1 + k * (( (k-1)**r - 1 ) // (k-2))
        else:
            moore = 1 + k * r
    else:
        r = (g - 3) // 2
        # k * (1 + (k-1) + ... + (k-1)^r)
        if k > 2:
            moore = k * ( ( (k-1)**(r+1) - 1 ) // (k-2) )
        else:
            moore = k * (r+1)
            
    # For k=4, g=10 (r=4): 1 + 4 * (1 + 3 + 9 + 27) = 1 + 4 * 40 = 161?
    # Wait. Formula for even g=2r:
    # n >= 1 + k * sum_{i=0}^{r-1} (k-1)^i
    # g=10 => r=5.
    # Sum i=0 to 4: 1 + 3 + 9 + 27 + 81 = 121.
    # n >= 1 + 4 * 121 = 485?
    # My previous calc was wrong.
    # Let's re-calculate Moore bound for k=4, g=10.
    # g=10, r=5.
    # Vertices at dist 0: 1
    # Dist 1: 4
    # Dist 2: 4*3 = 12
    # Dist 3: 4*3*3 = 36
    # Dist 4: 4*3^3 = 108
    # Total = 1 + 4 + 12 + 36 + 108 = 161.
    # Wait, the sum formula: sum_{i=0}^{r-1} (k-1)^i where r=g/2=5.
    # i=0: 1
    # i=1: 3
    # i=2: 9
    # i=3: 27
    # i=4: 81
    # Sum = 121.
    # n >= 1 + 4 * 121 = 485?
    # Why the discrepancy?
    # Dist 0: 1
    # Dist 1: k
    # Dist 2: k(k-1)
    # ...
    # Dist d: k(k-1)^{d-1}
    # For girth g=2r, we go up to distance r-1 without cycles closing.
    # Max distance is r-1.
    # So sum d=0 to r-1.
    # For g=10, r=5. Dist 0 to 4.
    # 1 + 4 + 12 + 36 + 108 = 161.
    # So Moore bound is 161.
    
    # Let's re-verify the formula.
    # n >= 1 + k * ( (k-1)^{g/2 - 1} - 1 ) / (k-2) ? No.
    # Sum_{i=0}^{r-1} (k-1)^i = ((k-1)^r - 1) / (k-2).
    # For r=5, k=4: (3^5 - 1)/2 = (243-1)/2 = 121.
    # 1 + 4*121 = 485.
    
    # Where is the error?
    # Dist 1: k
    # Dist 2: k(k-1)
    # Dist 3: k(k-1)^2
    # ...
    # Dist d: k(k-1)^{d-1}
    # Sum for d=1 to r-1 is k * Sum_{j=0}^{r-2} (k-1)^j.
    # Total n = 1 + k * Sum_{j=0}^{r-2} (k-1)^j.
    # For g=10, r=5. Sum j=0 to 3.
    # 1 + 3 + 9 + 27 = 40.
    # n = 1 + 4 * 40 = 161.
    
    # Okay, so Moore bound is 161.
    
    if (g - 2) % 2 == 0:
        r = g // 2
        # Sum j=0 to r-2
        if r > 1:
            if k > 2:
                moore = 1 + k * ( ((k-1)**(r-1) - 1) // (k-2) )
            else:
                moore = 1 + k * (r-1)
        else:
            moore = 1
    else:
        # Odd girth g=2r+1
        r = (g-1)//2
        # Sum j=0 to r-1
        if k > 2:
            moore = k * ( ((k-1)**r - 1) // (k-2) )
        else:
            moore = k * r

    # Fallback to seed algorithm if search fails
    def seed_construct():
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r_val, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r_val]

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

    # Search for smaller graph
    # Try n from moore to 350
    # For each n, try up to 100 random graphs
    
    def check_girth(n, edges, g_target):
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # Check for cycles shorter than g_target
        # BFS from each vertex
        for i in range(n):
            dist = [-1] * n
            dist[i] = 0
            q = deque([i])
            while q:
                u = q.popleft()
                if dist[u] >= g_target // 2:
                    break
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
                    elif dist[v] > dist[u] + 1:
                        # Cycle found? 
                        # If we encounter a visited node at a smaller distance, 
                        # it's a back edge. 
                        # But we need to be careful.
                        pass
            # This simple BFS doesn't detect girth directly easily.
            # Better: Count closed non-backtracking walks?
            # Or simply: for each edge (u,v), check if there is a path from v to u of length L < g_target-1 not using (u,v).
            
        # Efficient girth check:
        # For each vertex, BFS up to depth g//2.
        # If we find a back-edge to a node at depth d2 where d1+d2+1 < g, we have a cycle.
        for i in range(n):
            dist = [-1] * n
            dist[i] = 0
            parent = [-1] * n
            q = deque([i])
            while q:
                u = q.popleft()
                if dist[u] * 2 >= g_target:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        q.append(v)
                    else:
                        if v != parent[u]:
                            # Back edge
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < g_target:
                                return False
        return True

    def generate_random_4reg(n):
        # Configuration model
        half_edges = []
        for i in range(n):
            for _ in range(k):
                half_edges.append(i)
        random.shuffle(half_edges)
        edges = []
        for i in range(0, n*k, 2):
            u = half_edges[i]
            v = half_edges[i+1]
            if u == v:
                return None # Loop
            if (u, v) in edges or (v, u) in edges:
                return None # Multiedge
            edges.append((u, v))
        return edges

    # Search
    found = False
    best_n = None
    best_edges = None
    
    # Upper bound for search
    max_n = 400 
    if time.time() > deadline - 10:
        n, e = seed_construct()
        return n, e

    for n in range(int(moore), max_n):
        if time.time() > deadline - 10:
            break
        attempts = 50
        for _ in range(attempts):
            if time.time() > deadline - 10:
                break
            e = generate_random_4reg(n)
            if e is None:
                continue
            if check_girth(n, e, g):
                best_n = n
                best_edges = e
                found = True
                break
        if found:
            break

    if found:
        return best_n, best_edges
    else:
        # Fallback to seed
        n, e = seed_construct()
        return n, e
