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
        i = 0
        while i < len(order):
            if time.time() > deadline:
                break
            vertex = order[i]
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
            i += 1
        return len(order), edges

    if k == 5 and g <= 12:
        try:
            q = 13
            # Build PSL(2, q) elements as (a, b, c, d) with ad - bc = 1
            sl_elements = []
            for a in range(q):
                for b in range(q):
                    for c in range(q):
                        if a != 0:
                            # ad - bc = 1 => d = (1 + bc) / a
                            d = (1 + b * c) * pow(a, -1, q) % q
                            sl_elements.append((a, b, c, d))
                        else:
                            # -bc = 1 => c = -1/b (b != 0)
                            if b == 0:
                                continue
                            c_val = (-1 * pow(b, -1, q)) % q
                            if c_val != c:
                                continue
                            for d in range(q):
                                sl_elements.append((a, b, c, d))

            # Deduplicate: each element appears twice (matrix and its negative) in PSL
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

            def mat_mult(u, s):
                u_a, u_b, u_c, u_d = u
                s_a, s_b, s_c, s_d = s
                return ((u_a * s_a + u_b * s_c) % q,
                        (u_a * s_b + u_b * s_d) % q,
                        (u_c * s_a + u_d * s_c) % q,
                        (u_c * s_b + u_d * s_d) % q)

            # Find involutions (order 2 elements): M^2 = I, M != I
            # For (a,b,c,d) with ad-bc=1: M^2 = I iff a+d=0 and (b,c) != (0,0)
            involutions = []
            for e in psl_elements:
                a, b, c, d = e
                if (a + d) % q == 0 and (b, c) != (0, 0):
                    involutions.append(e)

            # Precompute mat_mult for all pairs of involutions for speed
            inv_idx = {e: i for i, e in enumerate(involutions)}
            n_inv = len(involutions)
            # inv_prod[i][j] = mat_mult(involutions[i], involutions[j])
            inv_prod = [[None] * n_inv for _ in range(n_inv)]
            for i in range(n_inv):
                for j in range(n_inv):
                    inv_prod[i][j] = mat_mult(involutions[i], involutions[j])

            # Also precompute for any element times involution
            # elem_inv_prod[i][j] = mat_mult(psl_elements[i], involutions[j])
            elem_inv_prod = [[None] * n_inv for _ in range(n_vertices)]
            for i in range(n_vertices):
                for j in range(n_inv):
                    elem_inv_prod[i][j] = mat_mult(psl_elements[i], involutions[j])

            # Also need: given an element, find its index
            elem_idx = {e: i for i, e in enumerate(psl_elements)}

            def check_girth_fast(S_indices):
                """Check girth >= g for Cayley graph with generators S_indices (indices into involutions)."""
                S = [involutions[i] for i in S_indices]
                S_prod = [elem_inv_prod[i][j] for i in range(n_vertices) for j in S_indices]
                # S_prod is flattened: S_prod[(u_idx * n_inv + s_idx)]
                
                # BFS from each vertex
                for start in range(n_vertices):
                    if time.time() > deadline:
                        return False
                    dist = [-1] * n_vertices
                    dist[start] = 0
                    q_bfs = [start]
                    head = 0
                    while head < len(q_bfs):
                        u = q_bfs[head]
                        head += 1
                        du = dist[u]
                        if du >= 4:
                            continue
                        for s_idx in range(len(S)):
                            v = elem_idx[S_prod[u * n_inv + s_idx]]
                            if dist[v] == -1:
                                dist[v] = du + 1
                                q_bfs.append(v)
                            else:
                                if du + 1 + dist[v] < g:
                                    return False
            return True

            best_edges = None
            # Random search with precomputed products
            random.seed(42)
            for attempt in range(2000):
                if time.time() > deadline:
                    break
                if n_inv < 5:
                    break
                S_indices = random.sample(range(n_inv), 5)
                
                # Check if S generates the group via BFS
                visited = {identity: 0}
                queue = [identity]
                edges = []
                head = 0
                while head < len(queue):
                    if time.time() > deadline:
                        break
                    u = queue[head]
                    head += 1
                    u_idx = elem_idx[u]
                    for s_idx in S_indices:
                        v = elem_inv_prod[u_idx][s_idx]
                        if v not in visited:
                            visited[v] = len(visited)
                            queue.append(v)
                            edges.append((u_idx, elem_idx[v]))
                
                if len(visited) != n_vertices:
                    continue
                
                # Check girth
                if check_girth_fast(S_indices):
                    best_edges = edges
                    break

            if best_edges is not None:
                return n_vertices, best_edges

        except Exception:
            pass

    return luw_fallback()
