import time
from collections import deque

def construct(k, g):
    t0 = time.time()
    deadline = 115.0
    
    def certify_girth(n, edges, g):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        N2 = n * n
        INF = g
        dist = [INF] * N2
        q = deque()
        
        for a in range(n):
            for b in adj[a]:
                dist[a * n + b] = 1
                q.append(a * n + b)
                
        while q:
            s = q.popleft()
            d = dist[s]
            if d >= g - 1:
                continue
            p = s // n
            c = s % n
            rev = c * n + p
            dr = dist[rev]
            if dr != INF:
                if d + dr + 1 < g:
                    return False
            for nb in adj[c]:
                if nb == p:
                    continue
                ns = c * n + nb
                nd = d + 1
                if nd < dist[ns]:
                    dist[ns] = nd
                    q.append(ns)
        return True

    def try_circulant(n, jumps):
        """Try a circulant graph with given jumps."""
        edges = []
        eset = set()
        for i in range(n):
            for j in jumps:
                v = (i + j) % n
                if i < v:
                    key = (i, v)
                    if key not in eset:
                        eset.add(key)
                        edges.append((i, v))
                elif v < i:
                    key = (v, i)
                    if key not in eset:
                        eset.add(key)
                        edges.append((v, i))
        # Check degree
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if any(x != k for x in deg):
            return None
        if len(eset) != n * k // 2:
            return None
        return n, edges

    best_n, best_edges = None, None
    
    if k == 3 and g == 16:
        # Try known good constructions
        # The (3,16)-cage has 936 vertices
        # Try circulant graphs with n=936
        
        n_target = 936
        # 936 = 2^3 * 3^2 * 13
        # Try various jump sets
        
        import random
        random.seed(42)
        
        # First, try some structured jump sets
        # For a 3-regular circulant, we need 3 jumps (or 2 if one is n/2)
        # n=936, n/2=468
        
        candidates = []
        
        # Try jumps that are "far apart" to avoid short cycles
        # A good heuristic: use jumps that are roughly n/3 apart
        base = n_target // 3
        for da in range(-20, 21):
            for db in range(-20, 21):
                a = (base + da) % n_target
                b = (2 * base + db) % n_target
                # Third jump: try n - a - b mod n, or some other
                c = (n_target - a - b) % n_target
                if a == 0 or b == 0 or c == 0:
                    continue
                if a == n_target - a or b == n_target - b or c == n_target - c:
                    continue  # Self-inverse, would give degree issues
                jumps = sorted([a, b, c])
                if jumps in candidates:
                    continue
                candidates.append(jumps)
        
        # Also try some random jump sets
        for _ in range(200):
            a = random.randint(1, n_target - 1)
            b = random.randint(1, n_target - 1)
            c = random.randint(1, n_target - 1)
            if a == n_target - a or b == n_target - b or c == n_target - c:
                continue
            jumps = sorted([a, b, c])
            if jumps not in candidates:
                candidates.append(jumps)
        
        for jumps in candidates:
            if time.time() - t0 > deadline - 10:
                break
            result = try_circulant(n_target, jumps)
            if result is None:
                continue
            n, edges = result
            if certify_girth(n, edges, g):
                best_n, best_edges = n, edges
                break
        
        # If circulant didn't work, try LUW as fallback
        if best_n is None and time.time() - t0 < deadline - 10:
            def luw_construct(k, g, d):
                q = k
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

            for d in [8, 9, 10]:
                if time.time() - t0 > deadline - 5:
                    break
                try:
                    n, edges = luw_construct(k, g, d)
                    if certify_girth(n, edges, g):
                        best_n, best_edges = n, edges
                        break
                except Exception:
                    continue
    
    if best_n is None:
        # Last resort: LUW with d=8
        def luw_construct(k, g, d):
            q = k
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

        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        best_n, best_edges = luw_construct(k, g, d)
    
    return best_n, best_edges
