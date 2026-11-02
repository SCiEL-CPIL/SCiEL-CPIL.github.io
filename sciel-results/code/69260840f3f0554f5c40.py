import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    # Fallback LUW construction (same as seed)
    def _luw_construct(k, g):
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

    # Fast girth check for 4-regular circulant C_n({s1, s2})
    def check_girth_4reg(n, s1, s2):
        # Build adjacency
        adj = [[] for _ in range(n)]
        for u in range(n):
            v1 = (u + s1) % n
            v2 = (u - s1) % n
            v3 = (u + s2) % n
            v4 = (u - s2) % n
            adj[u] = [v1, v2, v3, v4]
        
        # BFS from 0
        parent = [-1] * n
        depth = [-1] * n
        parent[0] = 0
        depth[0] = 0
        queue = [0]
        head = 0
        
        min_cycle = float('inf')
        
        while head < len(queue):
            u = queue[head]
            head += 1
            
            for v in adj[u]:
                if depth[v] == -1:
                    depth[v] = depth[u] + 1
                    parent[v] = u
                    queue.append(v)
                else:
                    if parent[u] == v or parent[v] == u:
                        continue
                    cycle_len = depth[u] + depth[v] + 1
                    if cycle_len < min_cycle:
                        min_cycle = cycle_len
                    if min_cycle < g:
                        return min_cycle
        return min_cycle

    # Search for 2-step circulant
    # n must be even? No, but 4-regular circulant on odd n is possible.
    # However, Moore bound is 243.
    # Try n from 244 to 318.
    for n in range(244, 320, 2):
        if time.time() > deadline:
            break
        
        # Sample random pairs of steps
        # Steps must be distinct and not n/2
        # Range 1 to n//2 - 1 (if n even) or n//2 (if n odd)
        # Let's just use 1 to n//2
        max_step = n // 2
        if n % 2 == 0:
            max_step -= 1 # Avoid n/2 which creates multi-edges or loops?
            # If s = n/2, u+s = u-s. Degree becomes 2 for that step?
            # We need 4 distinct neighbors.
            # So s != n/2.
        
        # Number of attempts
        attempts = 20000
        for _ in range(attempts):
            if time.time() > deadline:
                break
            
            # Pick 2 distinct steps
            s1 = random.randint(1, max_step)
            s2 = random.randint(1, max_step)
            if s1 == s2:
                continue
            
            # Optional: quick check for 4-cycles?
            # 4-cycle if 2*s1 == 2*s2 (mod n) -> s1 == s2 (handled)
            # or s1+s2 == 0 (mod n) -> s1 == -s2 (mod n) -> s1+s2 == n
            # or s1-s2 == 0 (handled)
            # So check s1 + s2 == n
            if s1 + s2 == n:
                continue
            
            girth = check_girth_4reg(n, s1, s2)
            if girth >= g:
                edges = []
                for u in range(n):
                    v1 = (u + s1) % n
                    v2 = (u - s1) % n
                    v3 = (u + s2) % n
                    v4 = (u - s2) % n
                    for v in [v1, v2, v3, v4]:
                        if u < v:
                            edges.append((u, v))
                return n, edges

    # Fallback
    return _luw_construct(k, g)
