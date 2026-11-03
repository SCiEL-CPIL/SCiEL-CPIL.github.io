import time
import random
import numpy as np
from itertools import combinations

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110

    def luw_fallback():
        # Keep the original fallback as is, but ensure it's fast
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

    if k == 5 and g == 9:
        try:
            q = 13
            # Build PSL(2, 13)
            sl_elements = []
            for a in range(q):
                for b in range(q):
                    for c in range(q):
                        if a != 0:
                            d_val = (1 + b * c) * pow(a, -1, q) % q
                            sl_elements.append((a, b, c, d_val))
                        else:
                            if b == 0:
                                continue
                            c_val = (-1 * pow(b, -1, q)) % q
                            if c_val != c:
                                continue
                            for d_val in range(q):
                                sl_elements.append((a, b, c, d_val))
            
            psl_elements = []
            seen = set()
            for e in sl_elements:
                a, b, c, d_val = e
                key = (a, b, c, d_val)
                neg_key = (-a % q, -b % q, -c % q, -d_val % q)
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
            
            # Multiplication table
            mul_table = np.zeros((n_vertices, n_vertices), dtype=np.int32)
            for i in range(n_vertices):
                a1, b1, c1, d1 = psl_elements[i]
                for j in range(n_vertices):
                    a2, b2, c2, d2 = psl_elements[j]
                    prod = ((a1 * a2 + b1 * c2) % q,
                            (a1 * b2 + b1 * d2) % q,
                            (c1 * a2 + d1 * c2) % q,
                            (c1 * b2 + d1 * d2) % q)
                    mul_table[i, j] = idx_map[prod]
            
            # Find involutions
            involution_indices = []
            for i, e in enumerate(psl_elements):
                a, b, c, d_val = e
                if (a + d_val) % q == 0 and (b, c) != (0, 0):
                    involution_indices.append(i)
            
            involution_indices = np.array(involution_indices, dtype=np.int32)
            n_inv = len(involution_indices)
            
            # Search for 5 involutions
            # Use a fixed set of random seeds for reproducibility
            found = False
            best_edges = None
            
            for seed in range(100):
                if time.time() > deadline - 10:
                    break
                np.random.seed(seed)
                # Try 1000 random sets per seed
                for _ in range(1000):
                    if time.time() > deadline - 10:
                        break
                    S_idx = np.random.choice(n_inv, size=5, replace=False)
                    psl_S = involution_indices[S_idx]
                    
                    # Check generation
                    visited = np.zeros(n_vertices, dtype=bool)
                    visited[0] = True
                    queue = [0]
                    head = 0
                    edges = []
                    while head < len(queue):
                        u = queue[head]
                        head += 1
                        for s in range(5):
                            v = mul_table[u, psl_S[s]]
                            if not visited[v]:
                                visited[v] = True
                                queue.append(v)
                                edges.append((u, v))
                    if not visited.all():
                        continue
                    
                    # Check girth
                    dist = np.full(n_vertices, -1, dtype=np.int32)
                    dist[0] = 0
                    queue_nb = [(0, -1)]
                    head_nb = 0
                    girth_ok = True
                    while head_nb < len(queue_nb):
                        u, prev_s = queue_nb[head_nb]
                        head_nb += 1
                        du = dist[u]
                        if du >= 4:
                            continue
                        for s in range(5):
                            if s == prev_s:
                                continue
                            v = mul_table[u, psl_S[s]]
                            dv = dist[v]
                            if dv == -1:
                                dist[v] = du + 1
                                queue_nb.append((v, s))
                            else:
                                if du + 1 + dv < g:
                                    girth_ok = False
                                    break
                        if not girth_ok:
                            break
                    if girth_ok:
                        best_edges = edges
                        found = True
                        break
                if found:
                    break
            
            if found:
                return n_vertices, best_edges
                
        except Exception:
            pass

    return luw_fallback()
