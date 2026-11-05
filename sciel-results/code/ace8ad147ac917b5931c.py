import time
import math
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140.0

    def remaining():
        return deadline - time.time()

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

        def is_prime(n):
            if n < 2:
                return False
            if n < 4:
                return True
            if n % 2 == 0 or n % 3 == 0:
                return False
            i = 5
            while i * i <= n:
                if n % i == 0 or n % (i + 2) == 0:
                    return False
                i += 6
            return True

        best_m = None
        best_x = None

        # Generate list of primes from min_m upward
        primes = []
        p = min_m
        while p < 2000:
            if is_prime(p):
                primes.append(p)
            p += 1

        for m in primes:
            if remaining() < 20:
                break
            if best_m is not None:
                break

            found = False

            # Family 1: quadratic v_i = (a*i^2 + b*i + c) % m
            max_a = min(m, 50)
            max_b = min(m, 50)
            max_c = min(m, 30)
            for a in range(1, max_a):
                if found:
                    break
                for b in range(max_b):
                    if found:
                        break
                    for c in range(max_c):
                        x = np.array([(a * i * i + b * i + c) % m for i in range(n_edges)])
                        if try_vector(x, m):
                            best_m, best_x = m, x
                            found = True
                            break

            if found:
                break

            # Family 2: linear v_i = (a*i + b) % m
            for a in range(1, min(m, 60)):
                if found:
                    break
                for b in range(min(m, 30)):
                    x = np.array([(a * i + b) % m for i in range(n_edges)])
                    if try_vector(x, m):
                        best_m, best_x = m, x
                        found = True
                        break

            if found:
                break

            # Family 3: exponential v_i = (a^i) % m
            for a in range(2, min(m, 60)):
                x = np.array([pow(a, i, m) for i in range(n_edges)])
                if try_vector(x, m):
                    best_m, best_x = m, x
                    found = True
                    break

            if found:
                break

            # Family 4: mixed v_i = (a*i^2 + b) % m
            for a in range(1, min(m, 40)):
                if found:
                    break
                for b in range(min(m, 40)):
                    x = np.array([(a * i * i + b) % m for i in range(n_edges)])
                    if try_vector(x, m):
                        best_m, best_x = m, x
                        found = True
                        break

            if found:
                break

            # Family 5: cubic v_i = (a*i^3 + b*i + c) % m
            for a in range(1, min(m, 20)):
                if found:
                    break
                for b in range(min(m, 20)):
                    if found:
                        break
                    for c in range(min(m, 15)):
                        x = np.array([(a * i * i * i + b * i + c) % m for i in range(n_edges)])
                        if try_vector(x, m):
                            best_m, best_x = m, x
                            found = True
                            break

            if found:
                break

            # Family 6: product form v_i = (a*i + b) * (c*i + d) % m
            for a in range(1, min(m, 25)):
                if found:
                    break
                for b in range(min(m, 25)):
                    if found:
                        break
                    for c in range(1, min(m, 25)):
                        if found:
                            break
                        for d in range(min(m, 25)):
                            x = np.array([((a * i + b) * (c * i + d)) % m for i in range(n_edges)])
                            if try_vector(x, m):
                                best_m, best_x = m, x
                                found = True
                                break

            if found:
                break

            # Random search with many restarts for this prime
            rng = np.random.default_rng(m * 2654435761 + 7)
            num_random = 3000
            for _ in range(num_random):
                if remaining() < 15:
                    break
                x = rng.integers(0, m, size=n_edges)
                if try_vector(x, m):
                    best_m, best_x = m, x
                    found = True
                    break

            # Simulated annealing polish: start from a random vector, try to fix zeros
            if not found and remaining() > 25:
                rng2 = np.random.default_rng(m * 999999 + 13)
                x = rng2.integers(0, m, size=n_edges)
                T = float(m)
                for step in range(5000):
                    if remaining() < 15:
                        break
                    S = (C @ x) % m
                    zero_mask = (S == 0)
                    if not np.any(zero_mask):
                        best_m, best_x = m, x
                        found = True
                        break
                    # Pick a random coordinate to perturb
                    idx = rng2.integers(0, n_edges)
                    old_val = x[idx]
                    x[idx] = rng2.integers(0, m)
                    S_new = (C @ x) % m
                    zero_count_new = np.sum(S_new == 0)
                    zero_count_old = np.sum(zero_mask)
                    if zero_count_new <= zero_count_old:
                        T *= 0.999
                    else:
                        # Accept with probability exp(-(delta)/T)
                        delta = zero_count_new - zero_count_old
                        if rng2.random() < math.exp(-delta / max(T, 1e-6)):
                            T *= 0.999
                        else:
                            x[idx] = old_val
                            T *= 0.999

        if best_m is None:
            # Extend to larger primes with more aggressive random search
            for m in range(2000, 5000):
                if remaining() < 10:
                    break
                if m < 2 or not is_prime(m):
                    continue
                rng = np.random.default_rng(m * 40503 + 7)
                for _ in range(5000):
                    if remaining() < 8:
                        break
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

    # Try the voltage lift first (best structural candidate for (3,16)).
    result = voltage_lift()
    if result is not None:
        return result

    # Fallback to the LUW construction.
    return luw_construct()
