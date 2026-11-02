import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s kill

    def luw_fallback():
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

    if k == 5 and g <= 12:
        try:
            q = 13
            sl_elements = []
            for a in range(q):
                for b in range(q):
                    for c in range(q):
                        if a != 0:
                            d = (1 + b * c) * pow(a, -1, q) % q
                            sl_elements.append((a, b, c, d))
                        else:
                            if b == 0:
                                continue
                            c_val = (-1 * pow(b, -1, q)) % q
                            if c_val != c:
                                continue
                            for d in range(q):
                                sl_elements.append((a, b, c, d))

            psl_elements = []
            seen = set()
            for e in sl_elements:
                a, b, c, d = e
                key = (a, b, c, d)
                neg_key = (-a % q, -b % q, -c % q, -d % q)
                if key < neg_key:
                    if key not in seen:
                        seen.add(key)
                        psl_elements.append(e)
                else:
                    if neg_key not in seen:
                        seen.add(neg_key)
                        psl_elements.append(neg_key)

            n_vertices = len(psl_elements)
            idx_map = {e: i for i, e in enumerate(psl_elements)}
            identity = (1, 0, 0, 1)

            # Precompute all elements as matrices for faster multiplication
            # Actually, let's just use the tuple representation and a helper
            def mat_mult(u, s):
                u_a, u_b, u_c, u_d = u
                s_a, s_b, s_c, s_d = s
                return ((u_a * s_a + u_b * s_c) % q,
                        (u_a * s_b + u_b * s_d) % q,
                        (u_c * s_a + u_d * s_c) % q,
                        (u_c * s_b + u_d * s_d) % q)

            # Find involutions
            involutions = []
            for e in psl_elements:
                a, b, c, d = e
                if (a + d) % q == 0 and (b, c) != (0, 0):
                    involutions.append(e)

            best_edges = None
            # We want to find a set S of 5 involutions such that the Cayley graph has girth >= g.
            # Girth >= 9 means no cycles of length 4, 6, 8.
            # In a Cayley graph, a cycle of length 4 corresponds to s1 s2 s1 s2 = I (or similar patterns).
            # A cycle of length 6 corresponds to s1 s2 s3 s1 s2 s3 = I.
            # A cycle of length 8 corresponds to s1 s2 s3 s4 s1 s2 s3 s4 = I.
            
            # Let's just do BFS girth check.
            # Optimization: Precompute neighbors for each element? No, S changes.
            
            # Random search
            for _ in range(500):
                if time.time() > deadline:
                    break
                if len(involutions) < 5:
                    break
                S = random.sample(involutions, 5)
                
                # Check if S generates the group
                visited = {identity: 0}
                queue = [identity]
                edges = []
                
                while queue:
                    if time.time() > deadline:
                        break
                    u = queue.pop(0)
                    for s in S:
                        v = mat_mult(u, s)
                        if v not in visited:
                            visited[v] = len(visited)
                            queue.append(v)
                            edges.append((idx_map[u], idx_map[v]))
                
                if len(visited) != n_vertices:
                    continue
                
                # Check girth
                girth_ok = True
                for start in range(n_vertices):
                    if time.time() > deadline:
                        girth_ok = False
                        break
                    dist = [-1] * n_vertices
                    dist[start] = 0
                    q_bfs = [start]
                    head = 0
                    while head < len(q_bfs):
                        u = q_bfs[head]
                        head += 1
                        if dist[u] >= 4:
                            continue
                        u_elem = psl_elements[u]
                        for s in S:
                            v = idx_map[mat_mult(u_elem, s)]
                            if dist[v] == -1:
                                dist[v] = dist[u] + 1
                                q_bfs.append(v)
                            else:
                                if dist[u] + 1 + dist[v] < 9:
                                    girth_ok = False
                                    break
                        if not girth_ok:
                            break
                    if not girth_ok:
                        break
                
                if girth_ok:
                    best_edges = edges
                    break

            if best_edges is not None:
                return n_vertices, best_edges

        except Exception:
            pass

    return luw_fallback()
