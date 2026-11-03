import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s kill

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

    # --- PSL2(q) Cayley graph approach for k=5, g=9 ---
    if k == 5 and g == 9:
        try:
            q = 13
            # Build PSL2(13) elements: 2x2 matrices mod 13 with det=1, modulo ±I
            sl_elements = []
            for a in range(q):
                for b in range(q):
                    for c in range(q):
                        # det = ad - bc = 1 mod q
                        if b == 0:
                            if c == 0:
                                continue
                            # ad = 1 + bc => d = (1+bc)*a^{-1}
                            # a must be nonzero
                            for a_val in range(1, q):
                                d_val = (1 + b * c) * pow(a_val, -1, q) % q
                                sl_elements.append((a_val, b, c, d_val))
                        else:
                            # ad = 1 + bc
                            for a_val in range(q):
                                d_val = (1 + b * c - a_val * c * pow(b, -1, q)) % q
                                # Wait, this isn't right. Let me redo.
                                # ad - bc = 1 => d = (1 + bc) * a^{-1} if a != 0
                                # If a = 0: -bc = 1 => bc = -1
                                pass
            # Redo: enumerate all (a,b,c,d) with ad-bc=1 mod q
            sl_elements = []
            for a in range(q):
                for b in range(q):
                    for c in range(q):
                        # ad = 1 + bc mod q
                        rhs = (1 + b * c) % q
                        if a == 0:
                            if rhs == 0:
                                # -bc = 1, so b*c = -1 mod q
                                # d is free? No: 0*d - bc = 1 => -bc = 1
                                # So we need bc = -1. d can be anything? No, the equation is 0*d - bc = 1.
                                # So -bc = 1 mod q, i.e., bc = q-1.
                                # d is not constrained by the determinant equation when a=0.
                                # But we need the matrix to be in SL2, so det=1.
                                # With a=0: det = 0*d - b*c = -bc. So -bc = 1 => bc = -1.
                                # d is free.
                                if (b * c) % q == (q - 1) % q:
                                    for d in range(q):
                                        sl_elements.append((0, b, c, d))
                            else:
                                pass
                        else:
                            d = rhs * pow(a, -1, q) % q
                            sl_elements.append((a, b, c, d))

            # Now quotient by ±I: identify M and -M
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
            identity_idx = idx_map[identity]

            # Precompute group multiplication table
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

            # Find all involutions (elements of order 2, i.e., M^2 = I, M != I)
            involution_indices = []
            for i, e in enumerate(psl_elements):
                if i == identity_idx:
                    continue
                a, b, c, d = e
                # M^2 = I => trace(M) = 0 (for SL2 over field)
                # Actually M^2 = I iff (a+d)^2 = 4 for SL2, but over finite fields:
                # M^2 = I iff M is conjugate to [[0,1],[-1,0]] or similar.
                # Simpler: just compute M^2 and check if it's identity.
                # M^2 = [[a^2+b^2, ab+bd], [ca+dc, cb+d^2]]
                # For this to be I: a^2+b^2=1, ab+bd=0, ca+dc=0, cb+d^2=1
                # From det=1: ad-bc=1.
                # Easier: just use the mul_table.
                if mul_table[i, i] == identity_idx:
                    involution_indices.append(i)

            involution_indices = np.array(involution_indices, dtype=np.int32)
            n_inv = len(involution_indices)

            # Precompute: for each involution, its neighbors (v * s for all v)
            # inv_neighbor[s_idx, v] = mul_table[v, involution_indices[s_idx]]
            # Shape: (n_inv, n_vertices)
            inv_neighbor = np.zeros((n_inv, n_vertices), dtype=np.int32)
            for s in range(n_inv):
                inv_neighbor[s, :] = mul_table[:, involution_indices[s]]

            def check_girth(S_indices):
                """Check if Cayley graph with generators S_indices has girth >= g.
                S_indices: array of 5 involution indices into involution_indices."""
                # Build neighbor table: neighbor[v, s] = inv_neighbor[S_indices[s], v]
                # Shape: (n_vertices, 5)
                neighbor = np.zeros((n_vertices, 5), dtype=np.int32)
                for s in range(5):
                    neighbor[:, s] = inv_neighbor[S_indices[s], :]

                # BFS from identity_idx, tracking parent to detect cycles
                dist = np.full(n_vertices, -1, dtype=np.int32)
                parent = np.full((n_vertices, 5), -1, dtype=np.int32)  # parent[v, s] = parent vertex via generator s
                dist[identity_idx] = 0
                current_level = np.array([identity_idx], dtype=np.int32)

                max_depth = g - 1  # We need no cycles of length < g, so BFS up to depth g-1
                for depth in range(max_depth):
                    if len(current_level) == 0:
                        break
                    if time.time() > deadline:
                        return False

                    # Expand all vertices in current_level
                    # For each vertex v in current_level, for each generator s, neighbor[v, s]
                    # Check if already visited
                    next_level_set = set()
                    for v in current_level:
                        for s in range(5):
                            u = neighbor[v, s]
                            if dist[u] == -1:
                                dist[u] = depth + 1
                                parent[u, s] = v
                                next_level_set.add(int(u))
                            else:
                                # Found a cycle
                                # Cycle length = dist[v] + 1 + dist[u]
                                # But we need to be careful: this might be a "backtrack"
                                # In a BFS tree, if u is already visited and u != parent of v,
                                # then we have a cycle of length dist[v] + 1 + dist[u]
                                cycle_len = dist[v] + 1 + dist[u]
                                if cycle_len < g:
                                    return False

                    current_level = np.array(list(next_level_set), dtype=np.int32)

                return True

            # Search for a valid set of 5 involutions
            # Strategy: try all combinations of 5 involutions
            # But that's C(n_inv, 5) which could be large. Use random sampling first.
            best_result = None

            # First, try a deterministic approach: pick involutions that are "spread out"
            # For PSL2(13), there are (q+1)/2 = 7 involutions per conjugacy class, and 2 classes?
            # Actually, in PSL2(13), the number of involutions is (q+1)/2 * (q-1)/2? Let's just count.
            # n_inv should be around 78 for q=13.

            # Try random subsets
            random.seed(42)
            tried = set()
            max_tries = 5000
            for attempt in range(max_tries):
                if time.time() > deadline:
                    break
                # Pick 5 distinct involutions
                S = np.random.choice(n_inv, size=5, replace=False)
                S = np.sort(S)
                key = tuple(S)
                if key in tried:
                    continue
                tried.add(key)

                if check_girth(S):
                    # Valid! Build the edge list
                    S_sorted = np.sort(S)
                    edges = []
                    for v in range(n_vertices):
                        for s in range(5):
                            u = int(inv_neighbor[S_sorted[s], v])
                            if v < u:
                                edges.append((v, u))
                    best_result = (n_vertices, edges)
                    break

            if best_result is not None:
                return best_result

        except Exception:
            pass

    return luw_fallback()
