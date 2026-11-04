import time
import random
from collections import deque
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115

    # --- LUW fallback (from seed) ---
    def _luw_graph(k, g):
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

    # --- Fast girth verification ---
    def _check_girth(adj, n, g_limit):
        max_depth = (g_limit - 1) // 2
        for start in range(n):
            dist = np.full(n, -1, dtype=np.int32)
            parent = np.full(n, -1, dtype=np.int32)
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                if dist[u] >= max_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        if dist[v] < max_depth:
                            queue.append(v)
                    elif v != parent[u] and u != parent[v]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g_limit:
                            return False
        return True

    # --- LPS Graph Construction ---
    def _lps_graph(q):
        """Construct LPS graph on SL(2, Z_q)."""
        # Generate all matrices in SL(2, Z_q)
        mats = []
        mat_to_idx = {}
        idx = 0
        for a in range(q):
            for b in range(q):
                for c in range(q):
                    # d = (1 - bc) / a mod q
                    # Need a invertible
                    if a == 0:
                        continue
                    # Find inverse of a
                    try:
                        a_inv = pow(a, -1, q)
                    except ValueError:
                        continue
                    for d in range(q):
                        if (a * d - b * c) % q == 1:
                            m = (a, b, c, d)
                            if m not in mat_to_idx:
                                mat_to_idx[m] = idx
                                mats.append(m)
                                idx += 1
        n = len(mats)
        
        # Define generators
        # g1 = [[1, 1], [0, 1]]
        # g2 = [[1, 0], [1, 1]]
        # g3 = [[0, -1], [1, 0]]
        gens = [
            (1, 1, 0, 1),
            (1, 0, 1, 1),
            (0, q-1, 1, 0) # -1 is q-1
        ]
        
        adj = [[] for _ in range(n)]
        edges = set()
        
        for i, m in enumerate(mats):
            for g in gens:
                # Multiply m * g
                a, b, c, d = m
                ga, gb, gc, gd = g
                na = (a*ga + b*gc) % q
                nb = (a*gb + b*gd) % q
                nc = (c*ga + d*gc) % q
                nd = (c*gb + d*gd) % q
                nm = (na, nb, nc, nd)
                if nm in mat_to_idx:
                    j = mat_to_idx[nm]
                    if i != j:
                        if j not in adj[i]:
                            adj[i].append(j)
                            adj[j].append(i)
                        if i < j:
                            edges.add((i, j))
        
        # Verify 3-regular
        for i in range(n):
            if len(adj[i]) != 3:
                return None, None
        return n, list(edges)

    # --- Main Logic ---
    if k == 3 and g == 16:
        # Try LPS graphs for small primes
        # q=7 -> N=336
        # q=11 -> N=1320
        # q=13 -> N=2184
        for q in [7, 11, 13, 17, 19]:
            if time.time() > deadline - 20:
                break
            n_lps, edges_lps = _lps_graph(q)
            if n_lps is not None:
                adj = [[] for _ in range(n_lps)]
                for u, v in edges_lps:
                    adj[u].append(v)
                    adj[v].append(u)
                if _check_girth(adj, n_lps, g):
                    return n_lps, edges_lps

        # Try Circulant (as in seed)
        moore_bound = 512
        n_luw, edges_luw = _luw_graph(k, g)
        
        # If LUW is already small, return it
        if n_luw < 1000:
            return n_luw, edges_luw

        # Search circulants
        max_n_try = 1000
        for n in range(moore_bound, max_n_try + 1, 2):
            if time.time() > deadline - 10:
                break
            if n % 2 != 0:
                continue
            a_candidates = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, n//4, n//3, n//5, n//7]
            tried = set()
            for a in a_candidates:
                if time.time() > deadline - 10:
                    break
                a = a % n
                if a <= 0 or a >= n or a == n // 2:
                    continue
                na = (n - a) % n
                if na == 0 or na == n // 2:
                    continue
                conn = [a, na, n // 2]
                conn_key = frozenset(conn)
                if conn_key in tried:
                    continue
                tried.add(conn_key)
                
                # Build and check
                adj = [[] for _ in range(n)]
                edges = []
                valid = True
                for i in range(n):
                    for c in conn:
                        j = (i + c) % n
                        if j not in adj[i]:
                            adj[i].append(j)
                            adj[j].append(i)
                            if i < j:
                                edges.append((i, j))
                for i in range(n):
                    if len(adj[i]) != 3:
                        valid = False
                        break
                if not valid:
                    continue
                if _check_girth(adj, n, g):
                    return n, edges

    # Fallback
    n_luw, edges_luw = _luw_graph(k, g)
    return n_luw, edges_luw
