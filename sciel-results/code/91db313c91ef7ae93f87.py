import time
import math
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0

    def remaining():
        return deadline - time.time()

    # ---------------- PSL(2,q) based construction ----------------
    def psl2q_construct():
        if k != 3 or g != 16:
            return None

        # PSL(2,q) for prime q or prime power
        # Elements are 2x2 matrices with det=1, modulo q
        # We use generators a, b that generate PSL(2,q)
        # The Cayley graph with generators {a, a^{-1}, b} is 3-regular if a != a^{-1}
        
        def mat_mult(A, B, mod):
            return [[(A[0][0]*B[0][0] + A[0][1]*B[1][0]) % mod,
                     (A[0][0]*B[0][1] + A[0][1]*B[1][1]) % mod],
                    [(A[1][0]*B[0][0] + A[1][1]*B[1][0]) % mod,
                     (A[1][0]*B[0][1] + A[1][1]*B[1][1]) % mod]]
        
        def mat_inv(A, mod):
            # For 2x2 with det=1: [[a,b],[c,d]]^-1 = [[d,-b],[-c,a]]
            return [[A[1][1] % mod, (-A[0][1]) % mod],
                    [(-A[1][0]) % mod, A[0][0] % mod]]
        
        def mat_eq(A, B):
            return A[0][0] == B[0][0] and A[0][1] == B[0][1] and A[1][0] == B[1][0] and A[1][1] == B[1][1]
        
        def mat_hash(A):
            return (A[0][0], A[0][1], A[1][0], A[1][1])
        
        # Try various prime q values
        # For PSL(2,q), the Cayley graph has |PSL(2,q)| = q(q^2-1)/2 vertices
        # We need this to be manageable and girth >= 16
        
        def is_prime(n):
            if n < 2:
                return False
            if n < 4:
                return True
            if n % 2 == 0:
                return False
            for i in range(3, int(math.isqrt(n)) + 1, 2):
                if n % i == 0:
                    return False
            return True
        
        # Start with small primes where |PSL(2,q)| is reasonable
        # q=7: 21, q=11: 66, q=13: 104, q=17: 240, q=19: 342, q=23: 600
        # q=29: 1218, q=31: 1495, q=37: 2592
        # Moore bound is 512, so we want q such that q(q^2-1)/2 >= 512
        # q=11: 66, q=13: 104, q=17: 240, q=19: 342, q=23: 600
        # So q=23 gives 600 vertices which is above Moore bound
        
        candidate_primes = [23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        
        for q in candidate_primes:
            if remaining() < 20:
                break
            if not is_prime(q):
                continue
            
            # |PSL(2,q)| = q(q^2-1)/2
            group_size = q * (q*q - 1) // 2
            
            # Try different generator pairs
            # Standard generators for PSL(2,q):
            # a = [[0,1],[-1,0]] (order 4)
            # b = [[1,1],[0,1]] (infinite order in SL, but finite in PSL)
            
            # In PSL(2,q), we identify M ~ -M
            # So we work in the projective plane: matrices up to scalar
            
            # Let's use a different approach: enumerate the group and build Cayley graph
            # Start with identity
            identity = [[1, 0], [0, 1]]
            
            # Generators in PSL(2,q):
            # g1 = [[0,1],[q-1,0]] which is -[[0,1],[1,0]], but in PSL we can use [[0,1],[1,0]]
            # Actually [[0,1],[1,0]] has det=-1, so in SL we need det=1
            # [[0,1],[q-1,0]] has det=1, and in PSL it's equivalent to [[0,1],[1,0]] since q-1 ≡ -1
            
            # Let's use:
            # A = [[0, 1], [q-1, 0]]  (this is -J where J=[[0,1],[1,0]], det(A)=1)
            # B = [[1, 1], [0, 1]]     (det=1)
            
            A = [[0, 1], [q-1, 0]]
            B = [[1, 1], [0, 1]]
            
            # In PSL(2,q), A has order 2 (since A^2 = [[q-1,0],[0,q-1]] = (q-1)*I ~ I in PSL)
            # B has infinite order in SL but finite in PSL
            
            # For a 3-regular Cayley graph, we need 3 generators (or 2 generators where one is its own inverse)
            # If A has order 2, then {A, B, B^{-1}} gives 3-regular
            
            # Let's enumerate the group using BFS
            # Represent elements as (a,b,c,d) where matrix is [[a,b],[c,d]], det=1, normalized
            
            def normalize(M, mod):
                # Normalize so that the first non-zero entry in first row is 1, or handle specially
                # For PSL, we identify M ~ -M, so we can normalize sign
                a, b, c, d = M[0][0] % mod, M[0][1] % mod, M[1][0] % mod, M[1][1] % mod
                # Ensure det = 1
                det = (a*d - b*c) % mod
                if det != 1:
                    # This shouldn't happen for group elements
                    pass
                # Normalize: if a != 0, divide by a (i.e., multiply by a^{-1})
                # But in PSL, M ~ -M, so we can flip sign
                # Let's just use the tuple as key, and treat (a,b,c,d) ~ (-a,-b,-c,-d)
                return (a, b, c, d)
            
            def elem_mult(e1, e2, mod):
                a1, b1, c1, d1 = e1
                a2, b2, c2, d2 = e2
                a = (a1*a2 + b1*c2) % mod
                b = (a1*b2 + b1*d2) % mod
                c = (c1*a2 + d1*c2) % mod
                d = (c1*b2 + d1*d2) % mod
                return (a, b, c, d)
            
            def elem_inv(e, mod):
                a, b, c, d = e
                # Inverse of [[a,b],[c,d]] with det=1 is [[d,-b],[-c,a]]
                return (d % mod, (-b) % mod, (-c) % mod, a % mod)
            
            # In PSL(2,q), (a,b,c,d) ~ (-a,-b,-c,-d)
            # We'll pick a canonical representative: make the first non-zero component positive
            def canonical(e, mod):
                a, b, c, d = e
                na, nb, nc, nd = (-a) % mod, (-b) % mod, (-c) % mod, (-d) % mod
                # Pick the one that comes first in some ordering
                # Let's say we prefer the one where the first non-zero entry is smaller
                if a != 0 or b != 0:
                    if a == 0 and b != 0:
                        # Compare b vs nb
                        if b < nb:
                            return e
                        else:
                            return (na, nb, nc, nd)
                    else:
                        if a < na:
                            return e
                        else:
                            return (na, nb, nc, nd)
                else:
                    # a=0, b=0, so c != 0 (det=1)
                    if c < nc:
                        return e
                    else:
                        return (na, nb, nc, nd)
            
            # Enumerate group
            gen_A = canonical(normalize(A, q), q)
            gen_B = canonical(normalize(B, q), q)
            gen_B_inv = canonical(elem_inv(gen_B, q), q)
            
            # Check that generators are distinct and not identity
            identity_canon = canonical((1, 0, 0, 1), q)
            
            if gen_A == identity_canon or gen_B == identity_canon or gen_B_inv == identity_canon:
                continue
            if gen_A == gen_B or gen_A == gen_B_inv or gen_B == gen_B_inv:
                continue
            
            # BFS to enumerate group
            group = {}
            group[identity_canon] = 0
            queue = deque([identity_canon])
            
            while queue:
                curr = queue.popleft()
                for gen in [gen_A, gen_B, gen_B_inv]:
                    nxt = canonical(elem_mult(curr, gen, q), q)
                    if nxt not in group:
                        group[nxt] = len(group)
                        queue.append(nxt)
            
            n_vertices = len(group)
            
            # Build edges
            edges = []
            edge_set = set()
            for elem, idx in group.items():
                for gen in [gen_A, gen_B, gen_B_inv]:
                    nxt = canonical(elem_mult(elem, gen, q), q)
                    if nxt in group:
                        nidx = group[nxt]
                        if idx < nidx:
                            edge_set.add((idx, nidx))
            
            edges = list(edge_set)
            
            # Verify degree
            adj = [[] for _ in range(n_vertices)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            # Check all vertices have degree 3
            valid = True
            for i in range(n_vertices):
                if len(adj[i]) != 3:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check girth
            gc = check_girth(n_vertices, edges, g, deadline - 10)
            if gc >= g:
                return (n_vertices, edges)
        
        return None

    # ---------------- fast girth verification (early exit) ----------------
    def check_girth(n, edges, g, time_limit):
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        min_cycle = float('inf')
        for s in range(n):
            if time.time() > time_limit:
                return -1
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
                            return min_cycle
                    else:
                        dist[nb] = dist[node] + 1
                        parent[nb] = node
                        queue.append(nb)
            if min_cycle < g:
                return min_cycle
        return min_cycle

    # ---------------- Voltage lift of the Petersen graph with polynomial search ----------------
    def voltage_lift():
        if k != 3 or g != 16:
            return None

        base_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
        ]
        edge_index = {}
        for i, (u, v) in enumerate(base_edges):
            edge_index[(u, v)] = i
            edge_index[(v, u)] = i

        adj = [[] for _ in range(10)]
        for u, v in base_edges:
            adj[u].append(v)
            adj[v].append(u)

        max_L = g - 1  # 15
        all_walk_vectors = set()

        for start_v in range(10):
            stack = [(start_v, -1, 0, [], [])]
            while stack:
                current, prev, length, p_edges, p_dirs = stack.pop()
                if length >= 5 and current == start_v:
                    vec = [0] * max_L
                    for e_idx, d in zip(p_edges, p_dirs):
                        vec[e_idx] += d
                    all_walk_vectors.add(tuple(vec))
                if length == max_L:
                    continue
                for next_v in adj[current]:
                    if next_v == prev:
                        continue
                    e_idx = edge_index[(current, next_v)]
                    u, v = base_edges[e_idx]
                    d = 1 if current == u else -1
                    stack.append((next_v, current, length + 1,
                                  p_edges + [e_idx], p_dirs + [d]))

        if not all_walk_vectors:
            return None

        C = np.array(list(all_walk_vectors), dtype=np.int64)
        n_edges = len(base_edges)  # 15

        # Moore bound for (3,16) is ~512, n = 10*m => m >= 52.
        min_m = 52

        def try_vector(x, m):
            S = (C @ x) % m
            return not np.any(S == 0)

        def find_prime_ge(n):
            if n <= 2:
                return 2
            cand = n if n % 2 == 1 else n + 1
            while True:
                if all(cand % i for i in range(3, int(math.isqrt(cand)) + 1, 2)):
                    return cand
                cand += 2

        # Systematic search over polynomial voltage assignments
        # v_i = (a*i^2 + b*i + c) % p for i = 0..14
        # and v_i = (a*i + b) % p, v_i = a^(i^2) % p, etc.
        best_m = None
        best_x = None

        primes = []
        p = min_m
        while p < 200:
            p = find_prime_ge(p)
            if p >= min_m:
                primes.append(p)
            p += 1

        for m in primes:
            if remaining() < 15:
                break
            if best_m is not None:
                break

            # Try various polynomial families
            found = False

            # Family 1: quadratic v_i = (a*i^2 + b*i + c) % m
            for a in range(1, min(m, 20)):
                if found:
                    break
                for b in range(min(m, 20)):
                    if found:
                        break
                    for c in range(min(m, 10)):
                        x = np.array([(a * i * i + b * i + c) % m for i in range(n_edges)])
                        if try_vector(x, m):
                            best_m, best_x = m, x
                            found = True
                            break

            if found:
                break

            # Family 2: linear v_i = (a*i + b) % m
            for a in range(1, min(m, 30)):
                if found:
                    break
                for b in range(min(m, 15)):
                    x = np.array([(a * i + b) % m for i in range(n_edges)])
                    if try_vector(x, m):
                        best_m, best_x = m, x
                        found = True
                        break

            if found:
                break

            # Family 3: exponential v_i = (a^i) % m
            for a in range(2, min(m, 30)):
                x = np.array([pow(a, i, m) for i in range(n_edges)])
                if try_vector(x, m):
                    best_m, best_x = m, x
                    found = True
                    break

            if found:
                break

            # Family 4: mixed v_i = (a*i^2 + b) % m
            for a in range(1, min(m, 15)):
                if found:
                    break
                for b in range(min(m, 15)):
                    x = np.array([(a * i * i + b) % m for i in range(n_edges)])
                    if try_vector(x, m):
                        best_m, best_x = m, x
                        found = True
                        break

            if found:
                break

            # Family 5: v_i = (a*i^3 + b*i + c) % m
            for a in range(1, min(m, 10)):
                if found:
                    break
                for b in range(min(m, 10)):
                    if found:
                        break
                    for c in range(min(m, 5)):
                        x = np.array([(a * i * i * i + b * i + c) % m for i in range(n_edges)])
                        if try_vector(x, m):
                            best_m, best_x = m, x
                            found = True
                            break

            if found:
                break

            # Random fallback for this prime
            rng = np.random.default_rng(m * 2654435761)
            for _ in range(200):
                x = rng.integers(0, m, size=n_edges)
                if try_vector(x, m):
                    best_m, best_x = m, x
                    found = True
                    break

        if best_m is None:
            # Extend to larger primes
            for m in range(200, 500):
                if remaining() < 10:
                    break
                if m < 2 or not all(m % i for i in range(2, int(math.isqrt(m)) + 1)):
                    continue
                rng = np.random.default_rng(m * 40503 + 7)
                for _ in range(400):
                    x = rng.integers(0, m, size=n_edges)
                    if try_vector(x, m):
                        best_m, best_x = m, x
                        break
                if best_m is not None:
                    break

        if best_m is None:
            return None

        # Construct the lifted graph
        n = 10 * best_m
        edges_set = set()
        for u, v in base_edges:
            e_idx = edge_index[(u, v)]
            shift = best_x[e_idx] % best_m
            for i in range(best_m):
                u_lift = u * best_m + i
                v_lift = v * best_m + (i + shift) % best_m
                if u_lift < v_lift:
                    edges_set.add((u_lift, v_lift))
                else:
                    edges_set.add((v_lift, u_lift))

        final_edges = list(edges_set)

        gc = check_girth(n, final_edges, g, deadline - 5)
        if gc == -1:
            return None
        if gc >= g:
            return (n, final_edges)
        return None

    # ---------------- LUW construction (guaranteed fallback) ----------------
    def _luw_ops(q):
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            return (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        return ((lambda x, y: (x + y) % q),
                (lambda x, y: (x - y) % q),
                (lambda x, y: x * y % q))

    def luw_construct():
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        add, sub, mul = _luw_ops(q)
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

    # Try PSL(2,q) construction first (non-Abelian group-based, structurally different)
    result = psl2q_construct()
    if result is not None:
        return result

    # Try the voltage lift as second option
    result = voltage_lift()
    if result is not None:
        return result

    # Fallback to the LUW construction.
    return luw_construct()
