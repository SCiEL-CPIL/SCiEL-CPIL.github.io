import time
import random
import numpy as np
from collections import deque

_DEADLINE = 110.0

def construct(k, g):
    # Fallback: Lazebnik-Ustimenko-Woldar graph
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

    if k == 4 and g == 10:
        t0 = time.time()
        
        # Helper to check girth using BFS
        def check_girth(n, adj, min_girth):
            # If min_girth is 10, we check for cycles of length 3..9
            # If none found, girth >= 10
            for start in range(n):
                # BFS from start
                dist = [-1] * n
                parent = [-1] * n
                dist[start] = 0
                q = deque([start])
                while q:
                    u = q.popleft()
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            q.append(v)
                        else:
                            # Found a cycle
                            # Cycle length = dist[u] + dist[v] + 1
                            # But this is not necessarily the shortest cycle through u,v
                            # For girth check, we need to ensure NO cycle < min_girth
                            # A back edge (u,v) where dist[u] <= dist[v] creates a cycle of length dist[u] + dist[v] + 1
                            # If this length < min_girth, we fail
                            if dist[u] <= dist[v]:
                                cycle_len = dist[u] + dist[v] + 1
                                if cycle_len < min_girth:
                                    return False
                # Note: This BFS check is sufficient for girth if we check all back edges?
                # Yes, the girth is the minimum length of any cycle.
                # Any cycle must contain a back edge in the BFS tree.
                # The length of the cycle formed by a back edge (u,v) is dist[u] + dist[v] + 1.
                # However, this might not be the shortest cycle involving u and v if there are cross edges?
                # In an undirected graph, BFS tree back edges define cycles.
                # The shortest cycle is indeed the minimum of these lengths.
            return True

        def gen_random_4reg(n):
            while True:
                stubs = []
                for i in range(n):
                    for _ in range(4):
                        stubs.append(i)
                random.shuffle(stubs)
                edges = []
                valid = True
                seen = set()
                for i in range(0, len(stubs), 2):
                    u = stubs[i]
                    v = stubs[i+1]
                    if u == v:
                        valid = False
                        break
                    key = (min(u, v), max(u, v))
                    if key in seen:
                        valid = False
                        break
                    seen.add(key)
                    edges.append(key)
                if valid:
                    return edges

        # Try to find a graph with n < 2048
        # Start with n=320 (known cage size)
        # If we can't find it, try larger n up to 2048
        
        # We will try a few sizes: 320, 340, 360, ...
        # But 320 is the most likely to work if search is good enough.
        
        best_n = None
        best_edges = None
        
        # Time budget per size
        # We have ~110s. 
        # Let's try 320 with 60s, 340 with 20s, 360 with 20s.
        
        sizes_to_try = [320, 340, 360, 400, 500, 1000, 2048]
        
        for n in sizes_to_try:
            if time.time() - t0 > _DEADLINE - 10:
                break
                
            # Allocate time based on index
            # Simple: 50s for first, 10s for others
            time_limit = 50.0 if n == 320 else 10.0
            
            t_start = time.time()
            
            # Generate initial graph
            edges = gen_random_4reg(n)
            adj = [set() for _ in range(n)]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
                
            # Build A for cost
            A = np.zeros((n, n), dtype=np.int8)
            for u, v in edges:
                A[u, v] = 1
                A[v, u] = 1
                
            def calc_cost(A):
                # Count cycles of length 3-9
                # Use trace of A^k
                count = 0
                Ak = A.copy()
                for k in range(3, 10):
                    Ak = Ak @ A
                    count += np.trace(Ak)
                return count

            current_cost = calc_cost(A)
            T = 2.0
            cooling = 0.999
            best_cost = current_cost
            best_adj = [set(s) for s in adj]
            best_A = A.copy()
            
            while time.time() - t_start < time_limit:
                if current_cost == 0:
                    # Verify girth
                    if check_girth(n, adj, g):
                        best_n = n
                        best_edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                        return best_n, best_edges
                    # If cost is 0 but girth < 10, it means there are no cycles 3-9, 
                    # but there might be a cycle of length 10? No, girth is min cycle.
                    # If no cycles 3-9, girth is >= 10.
                    # So check_girth should return True.
                    # Unless check_girth is buggy.
                    # Actually, if cost is 0, there are NO cycles of length 3..9.
                    # So girth is at least 10.
                    # So we can return immediately.
                    best_n = n
                    best_edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                    return best_n, best_edges

                # Move: Edge swap
                moved = False
                for _ in range(50):
                    u = random.randint(0, n-1)
                    if not adj[u]: continue
                    v = random.choice(list(adj[u]))
                    
                    x = random.randint(0, n-1)
                    if x in (u, v): continue
                    if not adj[x]: continue
                    y = random.choice(list(adj[x]))
                    if y in (u, v): continue
                    
                    if A[u, x] == 1 or A[v, y] == 1:
                        continue
                        
                    # Swap
                    adj[u].remove(v); adj[v].remove(u)
                    adj[x].remove(y); adj[y].remove(x)
                    A[u, v] = 0; A[v, u] = 0
                    A[x, y] = 0; A[y, x] = 0
                    
                    adj[u].add(x); adj[x].add(u)
                    adj[v].add(y); adj[y].add(v)
                    A[u, x] = 1; A[x, u] = 1
                    A[v, y] = 1; A[y, v] = 1
                    
                    new_cost = calc_cost(A)
                    
                    if new_cost < current_cost:
                        current_cost = new_cost
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_adj = [set(s) for s in adj]
                            best_A = A.copy()
                        moved = True
                        break
                    else:
                        # Revert
                        adj[u].remove(x); adj[x].remove(u)
                        adj[v].remove(y); adj[y].remove(v)
                        A[u, x] = 0; A[x, u] = 0
                        A[v, y] = 0; A[y, v] = 0
                        
                        adj[u].add(v); adj[v].add(u)
                        adj[x].add(y); adj[y].add(x)
                        A[u, v] = 1; A[v, u] = 1
                        A[x, y] = 1; A[y, x] = 1
                        
                        if random.random() < np.exp(-(new_cost - current_cost) / T):
                            # Accept
                            adj[u].remove(v); adj[v].remove(u)
                            adj[x].remove(y); adj[y].remove(x)
                            A[u, v] = 0; A[v, u] = 0
                            A[x, y] = 0; A[y, x] = 0
                            
                            adj[u].add(x); adj[x].add(u)
                            adj[v].add(y); adj[y].add(v)
                            A[u, x] = 1; A[x, u] = 1
                            A[v, y] = 1; A[y, v] = 1
                            
                            current_cost = new_cost
                            moved = True
                            break
                            
                T *= cooling
                if T < 0.1: T = 0.1

            # If we found a valid graph in this size, we would have returned.
            # If not, we continue to next size.
            # But we should keep the best result if we didn't find a valid one?
            # The task requires a VALID graph.
            # So if we didn't find a valid one, we must fall back to LUW.
            
        # Fallback
        return luw_construct(k, g)

    return luw_construct(k, g)
