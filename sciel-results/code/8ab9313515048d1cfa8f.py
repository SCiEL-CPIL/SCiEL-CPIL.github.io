import time
import math
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin before 150s kill

    def remaining():
        return deadline - time.time()

    # ---------- fast girth verification (early exit) ----------
    def check_girth(n, edges, g, time_limit):
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        for v in range(n):
            if len(adj[v]) != k:
                return -1
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

    # ---------- Petersen base graph ----------
    base_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),
    ]
    n_base = 10
    base_adj = [[] for _ in range(n_base)]
    for i, (u, v) in enumerate(base_edges):
        base_adj[u].append((v, i))
        base_adj[v].append((u, i))

    # ---------- closed non-backtracking walks in Petersen (length < g) ----------
    def _get_closed_nb_walks(g):
        walks = set()
        for start_v in range(n_base):
            queue = []
            for next_v, e_id in base_adj[start_v]:
                u, v = base_edges[e_id]
                sign = 1 if u == start_v else -1
                queue.append((next_v, e_id, 1, [(e_id, sign)]))
            while queue:
                curr_v, prev_e, length, path = queue.pop(0)
                if length >= g:
                    continue
                if curr_v == start_v and length >= 3:
                    walks.add(tuple(path))
                for next_v, e_id in base_adj[curr_v]:
                    if e_id == prev_e:
                        continue
                    u, v = base_edges[e_id]
                    sign = 1 if u == curr_v else -1
                    queue.append((next_v, e_id, length + 1, path + [(e_id, sign)]))
        return list(walks)

    closed_walks = _get_closed_nb_walks(g)
    n_edges_base = len(base_edges)
    walk_coeffs = []
    for w in closed_walks:
        coeffs = [0] * n_edges_base
        for e_id, sign in w:
            coeffs[e_id] += sign
        if any(c != 0 for c in coeffs):
            walk_coeffs.append(coeffs)

    # ---------- algebraic validity test for a voltage assignment ----------
    def _walks_ok(volts, m):
        for coeffs in walk_coeffs:
            s = 0
            for i, c in enumerate(coeffs):
                if c != 0:
                    s += c * volts[i]
            if s % m == 0:
                return False
        return True

    def _gcd_all(vals, m):
        g_val = m
        for v in vals:
            g_val = math.gcd(g_val, v)
        return g_val

    # ---------- build the 3-regular lift ----------
    def _build_lift(volts, m):
        lift_edges = []
        for i, (u, v) in enumerate(base_edges):
            vol = volts[i]
            for a in range(m):
                b = (a + vol) % m
                su = u * m + a
                sv = v * m + b
                if su < sv:
                    lift_edges.append((su, sv))
        return n_base * m, lift_edges

    def _search_lift(m, time_limit):
        gen = [a for a in range(1, m) if math.gcd(a, m) == 1]
        if not gen:
            return None
        # systematic: constant voltages, then simple patterns
        for a in gen:
            if time.time() > time_limit:
                return None
            if _gcd_all([a], m) != 1:
                continue
            if _walks_ok([a] * n_edges_base, m):
                return _build_lift([a] * n_edges_base, m)
        # simple mixed patterns
        L = len(gen)
        for idx in range(0, min(L * L, 400)):
            if time.time() > time_limit:
                return None
            p, q = idx // L, idx % L
            volts = [gen[(p + i * q) % L] for i in range(n_edges_base)]
            if _gcd_all(volts, m) != 1:
                continue
            if _walks_ok(volts, m):
                return _build_lift(volts, m)
        # random search
        rng = [a for a in range(1, m)]
        tries = 0
        while time.time() < time_limit and tries < 4000:
            tries += 1
            volts = [rng[(i * 31 + (i * i % 7)) % len(rng)] for i in range(n_edges_base)]
            if _gcd_all(volts, m) != 1:
                continue
            if _walks_ok(volts, m):
                return _build_lift(volts, m)
        return None

    # ---------- LUW construction (guaranteed-valid fallback) ----------
    def _luw_ops(q):
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            return (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        return ((lambda x, y: (x + y) % q),
                (lambda x, y: (x - y) % q),
                (lambda x, y: x * y % q))

    def generate_rules(d, variant=0):
        base = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        if variant == 0:
            for c in range(5, d + 1):
                r, c0 = (c - 5) % 4, c - (c - 5) % 4
                base[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
        elif variant == 1:
            for c in range(5, d + 1):
                base[c] = (1, c - 2) if c % 2 == 0 else (c - 1, 1)
        elif variant == 2:
            for c in range(5, d + 1):
                if c % 3 == 0:
                    base[c] = (1, c - 3)
                elif c % 3 == 1:
                    base[c] = (c - 1, 1)
                else:
                    base[c] = (c - 2, 1)
        elif variant == 3:
            for c in range(5, d + 1):
                base[c] = (1, c - (1 + c % 2))
        else:
            for c in range(5, d + 1):
                base[c] = [(1, c - 1), (c - 1, 1), (1, c - 2), (c - 2, 1)][c % 4]
        return base

    def build_luw(q, d, rule, time_limit):
        add, sub, mul = _luw_ops(q)

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
            if time.time() > time_limit:
                return None
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    def luw_construct(k, g):
        q = k
        d = max(3, (g - 2) // 2)
        best = None
        best_n = float('inf')
        for variant in range(5):
            if remaining() < 12:
                break
            rule = generate_rules(d, variant)
            res = build_luw(q, d, rule, deadline - 12)
            if res is None:
                continue
            n, edges = res
            if n >= best_n:
                continue
            gc = check_girth(n, edges, g, deadline - 8)
            if gc == -1:
                continue
            if gc >= g:
                best, best_n = (n, edges), n
        if best is None:
            for d_try in range(d + 1, d + 10):
                if remaining() < 10:
                    break
                for variant in range(3):
                    rule = generate_rules(d_try, variant)
                    res = build_luw(q, d_try, rule, deadline - 10)
                    if res is None:
                        continue
                    n, edges = res
                    gc = check_girth(n, edges, g, deadline - 5)
                    if gc == -1:
                        continue
                    if gc >= g:
                        best = (n, edges)
                        break
                if best is not None:
                    break
        if best is None:
            rule = generate_rules(d, 0)
            res = build_luw(q, d, rule, deadline - 2)
            if res is not None:
                best = res
        return best

    # ---------- main ----------
    if k != 3:
        res = luw_construct(k, g)
        return res if res is not None else (0, [])

    # k == 3: search Petersen voltage lifts (small n), fall back to LUW.
    best = None
    best_n = float('inf')

    moore_m = max(2, (512 + n_base - 1) // n_base)
    m_candidates = list(range(moore_m, moore_m + 60))
    m_candidates += list(range(moore_m + 60, moore_m + 300, 3))
    m_candidates += list(range(moore_m + 300, moore_m + 900, 10))

    for m in m_candidates:
        if remaining() < 15:
            break
        if n_base * m >= best_n:
            continue
        res = _search_lift(m, deadline - 12)
        if res is None:
            continue
        n, edges = res
        if n >= best_n:
            continue
        gc = check_girth(n, edges, g, deadline - 8)
        if gc == -1:
            continue
        if gc >= g:
            best, best_n = (n, edges), n

    if best is not None:
        return best

    # fall back to LUW (guaranteed valid)
    res = luw_construct(k, g)
    return res if res is not None else (0, [])
