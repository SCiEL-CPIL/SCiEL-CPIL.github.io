import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    def check_girth(n, edges, g):
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        
        min_cycle = float('inf')
        for s in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            while qi < len(queue):
                node = queue[qi]
                qi += 1
                for nb in adj[node]:
                    if nb == parent[node]:
                        continue
                    if dist[nb] != -1:
                        cycle_len = dist[node] + dist[nb] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if cycle_len < g:
                            return False
                    else:
                        dist[nb] = dist[node] + 1
                        parent[nb] = node
                        queue.append(nb)
            if min_cycle < g:
                return False
        return True

    def luw_construct(k, g, d, rule_type=0):
        q = k
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q

        # Generate rule
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        if rule_type == 0:
            # Standard from seed
            for c in range(5, d + 1):
                r, c0 = (c - 5) % 4, c - (c - 5) % 4
                rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        elif rule_type == 1:
            # Shifted
            for c in range(5, d + 1):
                r, c0 = (c - 4) % 4, c - (c - 4) % 4
                rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        else:
            # Random-ish deterministic
            for c in range(5, d + 1):
                r = (c * 7) % 4
                c0 = c
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
        index = {start: 0}
        order = [start]
        edges = []

        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))

        n = len(order)
        return n, edges

    if k != 3:
        # Fallback for other k
        d = max(3, (g - 2) // 2)
        n, edges = luw_construct(k, g, d, 0)
        if check_girth(n, edges, g):
            return n, edges
        # If fails, try larger d
        for d_try in range(d + 1, d + 5):
            if time.time() > deadline - 10:
                break
            n, edges = luw_construct(k, g, d_try, 0)
            if check_girth(n, edges, g):
                return n, edges
        return n, edges

    # For k=3, try to find a smaller d with a good rule
    # Target girth is g.
    # Try d = (g-2)//2 - 1 first?
    # Standard d = (g-2)//2.
    d_std = max(3, (g - 2) // 2)
    
    best_n = float('inf')
    best_edges = []
    
    # Try d_std - 1 with various rules
    d_small = d_std - 1
    if d_small >= 3:
        for rt in range(3):
            if time.time() > deadline - 20:
                break
            n, edges = luw_construct(3, g, d_small, rt)
            if check_girth(n, edges, g):
                if n < best_n:
                    best_n = n
                    best_edges = edges
                break # Found one for this d, no need to try other rules for same d? 
                       # Actually, different rules might yield different n? No, n is 2*q^d.
                       # So if one works, n is fixed.
    
    # If no small d worked, try d_std with standard rule
    if best_n == float('inf'):
        for rt in range(3):
            if time.time() > deadline - 20:
                break
            n, edges = luw_construct(3, g, d_std, rt)
            if check_girth(n, edges, g):
                if n < best_n:
                    best_n = n
                    best_edges = edges
                break

    # If still nothing, try d_std + 1
    if best_n == float('inf'):
        d_large = d_std + 1
        n, edges = luw_construct(3, g, d_large, 0)
        if check_girth(n, edges, g):
            best_n = n
            best_edges = edges

    if best_n != float('inf'):
        return best_n, best_edges
    
    # Fallback
    n, edges = luw_construct(3, g, d_std, 0)
    return n, edges
