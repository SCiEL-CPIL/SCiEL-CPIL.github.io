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

    if k == 5 and g <= 12:
        try:
            q = 13
            # Build PSL(2,13) elements: matrices [a b; c d] with det = 1, mod q
            psl_elements = []
            seen = set()
            for a in range(q):
                for b in range(q):
                    for c in range(q):
                        if a != 0:
                            d = (1 + b * c) * pow(a, -1, q) % q
                            mat = (a, b, c, d)
                            neg = (-a % q, -b % q, -c % q, -d % q)
                            canon = mat if mat < neg else neg
                            if canon not in seen:
                                seen.add(canon)
                                psl_elements.append(canon)
                        else:
                            if b == 0:
                                continue
                            c_val = (-1 * pow(b, -1, q)) % q
                            if c_val != c:
                                continue
                            for d in range(q):
                                mat = (a, b, c, d)
                                neg = (-a % q, -b % q, -c % q, -d % q)
                                canon = mat if mat < neg else neg
                                if canon not in seen:
                                    seen.add(canon)
                                    psl_elements.append(canon)

            n_vertices = len(psl_elements)
            idx_map = {e: i for i, e in enumerate(psl_elements)}
            identity = (1, 0, 0, 1)

            def mat_mult(u, s):
                ua, ub, uc, ud = u
                sa, sb, sc, sd = s
                return ((ua * sa + ub * sc) % q,
                        (ua * sb + ub * sd) % q,
                        (uc * sa + ud * sc) % q,
                        (uc * sb + ud * sd) % q)

            # Find involutions: elements of order 2
            involutions = []
            for e in psl_elements:
                a, b, c, d = e
                if (a + d) % q == 0 and (b, c) != (0, 0):
                    m2 = mat_mult(e, e)
                    if m2 == identity:
                        involutions.append(e)

            n_inv = len(involutions)

            # Precompute prod_table[u_idx][j] = index of (psl_elements[u_idx] * involutions[j])
            prod_table = np.empty((n_vertices, n_inv), dtype=np.int32)
            for u_idx, u in enumerate(psl_elements):
                for j, s in enumerate(involutions):
                    prod_table[u_idx, j] = idx_map[mat_mult(u, s)]

            # Precompute pair products for 4-cycle check
            pair_is_inv = np.zeros((n_inv, n_inv), dtype=bool)
            for i in range(n_inv):
                for j in range(i + 1, n_inv):
                    prod = mat_mult(involutions[i], involutions[j])
                    prod2 = mat_mult(prod, prod)
                    if prod2 == identity:
                        pair_is_inv[i, j] = True
                        pair_is_inv[j, i] = True

            # Precompute triple products for 6-cycle check
            triple_is_inv = np.zeros((n_inv, n_inv, n_inv), dtype=bool)
            for i in range(n_inv):
                for j in range(n_inv):
                    if i == j:
                        continue
                    ij = mat_mult(involutions[i], involutions[j])
                    for l in range(n_inv):
                        if l == i or l == j:
                            continue
                        ijl = mat_mult(ij, involutions[l])
                        ijl2 = mat_mult(ijl, ijl)
                        if ijl2 == identity:
                            triple_is_inv[i, j, l] = True

            def check_girth_nb(S_indices):
                """Check girth >= g using non-backtracking walk enumeration from vertex 0.
                Since the graph is vertex-transitive, this suffices."""
                S = list(S_indices)
                
                if time.time() > deadline - 2:
                    return False
                
                # Level-by-level BFS in the group
                current_level = {(identity, -1)}
                
                for length in range(1, g):
                    if time.time() > deadline - 1:
                        return False
                    next_level = set()
                    for prod, last in current_level:
                        for s_idx in range(5):
                            if s_idx == last:
                                continue
                            new_prod = mat_mult(prod, involutions[S[s_idx]])
                            next_level.add((new_prod, s_idx))
                    
                    # Check for closed walks of length >= 3
                    if length >= 3:
                        for prod, last in next_level:
                            if prod == identity:
                                return False
                    
                    current_level = next_level
                    if not current_level:
                        break
                
                return True

            # Search for good S
            random.seed(42)
            
            # Phase 1: Random search with quick rejection
            max_trials = 30000
            for trial in range(max_trials):
                if time.time() > deadline - 10:
                    break
                S_indices = random.sample(range(n_inv), 5)

                # Quick 4-cycle check
                ok = True
                for i in range(5):
                    for j in range(i + 1, 5):
                        if pair_is_inv[S_indices[i]][S_indices[j]]:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue

                # Quick 6-cycle check
                ok = True
                for i in range(5):
                    for j in range(i + 1, 5):
                        for l in range(j + 1, 5):
                            if triple_is_inv[S_indices[i]][S_indices[j]][S_indices[l]]:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                if not ok:
                    continue

                # Full girth check via non-backtracking walk enumeration
                if check_girth_nb(S_indices):
                    edges = []
                    for u in range(n_vertices):
                        for si in S_indices:
                            v = prod_table[u, si]
                            if u < v:
                                edges.append((u, v))
                    return n_vertices, edges

            # Phase 2: Systematic search with local optimization
            for attempt in range(1000):
                if time.time() > deadline - 5:
                    break
                # Generate a set that passes 4-cycle check
                S = None
                for _ in range(200):
                    S = random.sample(range(n_inv), 5)
                    if all(not pair_is_inv[S[i]][S[j]] for i in range(5) for j in range(i + 1, 5)):
                        break
                if S is None:
                    continue

                # Try local search: swap one element at a time
                for _ in range(80):
                    if time.time() > deadline - 5:
                        break
                    if check_girth_nb(S):
                        edges = []
                        for u in range(n_vertices):
                            for si in S:
                                v = prod_table[u, si]
                                if u < v:
                                    edges.append((u, v))
                        return n_vertices, edges

                    # Try swapping one element
                    pos = random.randint(0, 4)
                    old = S[pos]
                    candidates = [x for x in range(n_inv) if x not in S]
                    random.shuffle(candidates)
                    for new in candidates[:50]:
                        S[pos] = new
                        # Quick check
                        ok = True
                        for i in range(5):
                            for j in range(i + 1, 5):
                                if pair_is_inv[S[i]][S[j]]:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if ok:
                            ok = True
                            for i in range(5):
                                for j in range(i + 1, 5):
                                    for l in range(j + 1, 5):
                                        if triple_is_inv[S[i]][S[j]][S[l]]:
                                            ok = False
                                            break
                                    if not ok:
                                        break
                                if not ok:
                                    break
                        if ok and check_girth_nb(S):
                            edges = []
                            for u in range(n_vertices):
                                for si in S:
                                    v = prod_table[u, si]
                                    if u < v:
                                        edges.append((u, v))
                            return n_vertices, edges
                        S[pos] = old

        except Exception:
            pass

    return luw_fallback()
