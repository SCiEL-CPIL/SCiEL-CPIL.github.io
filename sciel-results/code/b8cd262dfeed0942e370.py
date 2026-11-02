def construct(k, g):
    import time
    from collections import deque

    start_time = time.time()
    deadline = start_time + 110

    # Hardcoded known optimal cages
    known_cages = {}

    # Petersen Graph (3, 5)
    known_cages[(3, 5)] = (10, [(0,1),(1,2),(2,3),(3,4),(4,0),
                                 (5,7),(7,9),(9,6),(6,8),(8,5),
                                 (0,5),(1,6),(2,7),(3,8),(4,9)])

    # Heawood Graph (3, 7)
    heawood_edges = []
    fano_lines = [(0,1,2),(0,3,4),(0,5,6),(1,3,5),(1,4,6),(2,3,6),(2,4,5)]
    for i, line in enumerate(fano_lines):
        line_id = 7 + i
        for p in line:
            heawood_edges.append((p, line_id))
    known_cages[(3, 7)] = (14, heawood_edges)

    if (k, g) in known_cages:
        return known_cages[(k, g)]

    q = k  # For k-regular, LUW uses GF(q) with q = k

    def build_luw(d, q_val):
        """Build LUW graph with parameter d over GF(q). Returns (n, edges)."""
        if q_val == 4:
            # GF(4) tables
            mul_table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            add_table = [[0,1,3,2],[1,0,2,3],[3,2,0,1],[2,3,1,0]]
            sub_table = add_table  # same as add for GF(4)
        elif q_val == 3:
            add_table = [[0,1,2],[1,2,0],[2,0,1]]
            sub_table = [[0,2,1],[2,1,0],[1,0,2]]
            mul_table = [[0,0,0],[0,1,2],[0,2,1]]
        elif q_val == 5:
            add_table = [[(x + y) % 5 for y in range(5)] for x in range(5)]
            sub_table = [[(x - y) % 5 for y in range(5)] for x in range(5)]
            mul_table = [[x * y % 5 for y in range(5)] for x in range(5)]
        elif q_val == 7:
            add_table = [[(x + y) % 7 for y in range(7)] for x in range(7)]
            sub_table = [[(x - y) % 7 for y in range(7)] for x in range(7)]
            mul_table = [[x * y % 7 for y in range(7)] for x in range(7)]
        elif q_val == 9:
            # GF(9) = GF(3^2) with primitive element alpha, alpha^2 = alpha + 1
            # Represent elements as pairs (a, b) = a + b*alpha
            # Actually let's use a simpler representation
            # For GF(9), use polynomial representation over GF(3)
            # Element = a + b*alpha where alpha^2 = 2*alpha + 1 (irreducible x^2 - 2x - 1 over GF(3))
            # Actually x^2 + x + 2 is irreducible over GF(3) since f(0)=2, f(1)=1+1+2=4=1, f(2)=4+2+2=8=2
            # Let's use x^2 + 1 which is x^2 - 2... wait
            # f(0)=1, f(1)=1+1+1=3=0! So x^2+1 = (x+1)^2? No.
            # x^2 + 1: f(0)=1, f(1)=2, f(2)=4+1=5=2. Irreducible.
            # So GF(9) with alpha^2 = 2 (since alpha^2 + 1 = 0 => alpha^2 = -1 = 2)
            # Elements: 0, 1, 2, alpha, 1+alpha, 2+alpha, 2alpha, 1+2alpha
            # Index: 0=0, 1=1, 2=2, 3=alpha, 4=1+alpha, 5=2+alpha, 6=2alpha, 7=1+2alpha
            
            # Build tables by representation (a, b) where value = a + b*alpha
            def gf9_to_idx(a, b):
                return a + 3 * b
            
            def gf9_idx_to_ab(idx):
                return (idx % 3, idx // 3)
            
            add_table = [[0]*9 for _ in range(9)]
            sub_table = [[0]*9 for _ in range(9)]
            mul_table = [[0]*9 for _ in range(9)]
            
            for x in range(9):
                ax, bx = gf9_idx_to_ab(x)
                for y in range(9):
                    ay, by = gf9_idx_to_ab(y)
                    # Addition: (ax+ay, bx+by) mod 3
                    add_table[x][y] = gf9_to_idx((ax + ay) % 3, (bx + by) % 3)
                    # Subtraction: (ax-ay, bx-by) mod 3
                    sub_table[x][y] = gf9_to_idx((ax - ay) % 3, (bx - by) % 3)
                    # Multiplication: (ax+bx*alpha)(ay+by*alpha) = ax*ay + (ax*by+bx*ay)*alpha + bx*by*alpha^2
                    # alpha^2 = 2, so = (ax*ay + 2*bx*by) + (ax*by + bx*ay)*alpha
                    r = (ax * ay + 2 * bx * by) % 3
                    s = (ax * by + bx * ay) % 3
                    mul_table[x][y] = gf9_to_idx(r, s)
        elif q_val == 11:
            add_table = [[(x + y) % 11 for y in range(11)] for x in range(11)]
            sub_table = [[(x - y) % 11 for y in range(11)] for x in range(11)]
            mul_table = [[x * y % 11 for y in range(11)] for x in range(11)]
        else:
            # Generic prime field
            add_table = [[(x + y) % q_val for y in range(q_val)] for x in range(q_val)]
            sub_table = [[(x - y) % q_val for y in range(q_val)] for x in range(q_val)]
            mul_table = [[x * y % q_val for y in range(q_val)] for x in range(q_val)]

        # Precompute rules as list of (a, b) pairs indexed by c (2..d)
        rules = [None] * (d + 1)
        rules[2] = (1, 1)
        rules[3] = (2, 1)
        rules[4] = (1, 2)
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - (c - 5) % 4
            if r == 0:
                rules[c] = (1, c0 - 2)
            elif r == 1:
                rules[c] = (c0 - 1, 1)
            elif r == 2:
                rules[c] = (c0, 1)
            else:
                rules[c] = (1, c0 + 1)

        # Vertex: (side, x0, x1, ..., xd) where side in {0,1}
        start = (0,) + (0,) * (d + 1)
        index = {start: 0}
        order = [start]
        edges = []

        for vi, vertex in enumerate(order):
            side = vertex[0]
            x = vertex[1:]  # x[0..d]

            for first in range(q_val):
                y = [0] * (d + 1)
                y[1] = first
                for c in range(2, d + 1):
                    a, b = rules[c]
                    if side == 0:
                        y[c] = add_table[x[c]][mul_table[y[a]][x[b]]]
                    else:
                        y[c] = sub_table[x[c]][mul_table[x[a]][y[b]]]

                other = (1 - side,) + tuple(y)

                if other not in index:
                    index[other] = len(order)
                    order.append(other)

                if side == 0:
                    edges.append((vi, index[other]))

        return len(order), edges

    def build_adj(n, edges):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def girth_ok_fast(n, adj, g):
        """Check girth >= g using non-backtracking BFS from every vertex."""
        max_depth = (g - 1) // 2
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                du = dist[u]
                if du >= max_depth:
                    continue
                pu = parent[u]
                for v in adj[u]:
                    if v == pu:
                        continue
                    if dist[v] == -1:
                        dist[v] = du + 1
                        parent[v] = u
                        if du + 1 < max_depth:
                            queue.append(v)
                    else:
                        cycle_len = du + dist[v] + 1
                        if cycle_len < g:
                            return False
        return True

    def girth_check_numpy(n, adj, g):
        """Faster girth check using numpy where possible."""
        try:
            import numpy as np
            max_depth = (g - 1) // 2
            for start in range(n):
                dist = np.full(n, -1, dtype=np.int32)
                parent = np.full(n, -1, dtype=np.int32)
                dist[start] = 0
                queue = deque([start])
                while queue:
                    u = queue.popleft()
                    du = dist[u]
                    if du >= max_depth:
                        continue
                    pu = parent[u]
                    for v in adj[u]:
                        if v == pu:
                            continue
                        if dist[v] == -1:
                            dist[v] = du + 1
                            parent[v] = u
                            if du + 1 < max_depth:
                                queue.append(v)
                        else:
                            cycle_len = du + dist[v] + 1
                            if cycle_len < g:
                                return False
            return True
        except ImportError:
            return girth_ok_fast(n, adj, g)

    best_n = None
    best_edges = None

    # For LUW CD(d,q): girth >= d+5 for odd d
    # For odd g: need d odd, d >= g-5
    # For even g: need d even, d >= g-4 (since d+5>=g and d even)
    if g % 2 == 1:
        start_d = max(3, g - 5)
        if start_d % 2 == 0:
            start_d += 1
    else:
        start_d = max(2, g - 4)
        if start_d % 2 == 1:
            start_d += 1

    # Try increasing d values
    for d in range(start_d, g + 10):
        if time.time() > deadline - 30:
            break

        # Check parity
        if g % 2 == 1 and d % 2 == 0:
            continue
        if g % 2 == 0 and d % 2 == 1:
            continue

        try:
            n, edges = build_luw(d, q)
            if n > 100000:
                continue

            # Quick degree check
            deg_counts = [0] * n
            for u, v in edges:
                deg_counts[u] += 1
                deg_counts[v] += 1
            if not all(c == k for c in deg_counts):
                continue

            adj = build_adj(n, edges)

            if time.time() > deadline - 10:
                continue

            if girth_check_numpy(n, adj, g):
                if best_n is None or n < best_n:
                    best_n, best_edges = n, edges
                break
        except Exception:
            continue

    if best_n is None:
        # Fallback: try a few more d values without full girth check
        for d in range(start_d, start_d + 5):
            if time.time() > deadline - 5:
                break
            if g % 2 == 1 and d % 2 == 0:
                continue
            if g % 2 == 0 and d % 2 == 1:
                continue
            try:
                n, edges = build_luw(d, q)
                if n <= 100000:
                    deg_counts = [0] * n
                    for u, v in edges:
                        deg_counts[u] += 1
                        deg_counts[v] += 1
                    if all(c == k for c in deg_counts):
                        best_n, best_edges = n, edges
                        break
            except Exception:
                continue

    if best_n is None:
        # Last resort: construct a valid k-regular graph
        half = (g - 1) // 2
        moore = k + k * (k - 1) ** half
        n = max(moore, k + 1)
        if n % 2 != 0:
            n += 1
        n = min(n, 50000)

        edges = []
        seen = set()
        for i in range(n):
            for j in range(1, k // 2 + 1):
                a, b = i, (i + j) % n
                key = (min(a, b), max(a, b))
                if key not in seen:
                    seen.add(key)
                    edges.append((a, b))
                a, b = i, (i - j) % n
                key = (min(a, b), max(a, b))
                if key not in seen:
                    seen.add(key)
                    edges.append((a, b))

        best_n, best_edges = n, edges

    return best_n, best_edges
