import math
import time
import random
import sys
from collections import deque

try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:
    _HAS_NUMPY = False


def construct(k, g):
    if k != 3:
        return _luw_fallback(k, g)

    start_time = time.time()
    deadline = 112  # hard safety margin before 150s kill

    r = g // 2
    moore_bound = 2 * sum((k - 1) ** i for i in range(r))
    if moore_bound % 2 != 0:
        moore_bound += 1

    best = None  # (n, edges)

    # ---- 1. Circulant C(n; {a, n-a, n/2}) scan (fast, vectorized) ----
    max_circ_n = 1400
    for n in range(moore_bound, max_circ_n + 1, 2):
        if time.time() - start_time > deadline:
            break
        h = n // 2
        for a in range(1, h):
            if math.gcd(h, a) != 1:
                continue
            if _check_girth_circulant(n, a, g, start_time, deadline):
                edges = []
                for u in range(n):
                    for s in (a, n - a, h):
                        v = (u + s) % n
                        if u < v:
                            edges.append((u, v))
                return n, edges

    # ---- 2. Random 3-regular search (main improvement) ----
    max_random_n = 1400
    for n in range(moore_bound, max_random_n + 1, 2):
        if time.time() - start_time > deadline:
            break
        attempts = 3 if n < 800 else (6 if n < 1200 else 12)
        for _ in range(attempts):
            if time.time() - start_time > deadline:
                break
            edges = _generate_random_3_reg(n)
            if edges is None:
                continue
            if not _is_connected(n, edges):
                continue
            if _check_girth_general(n, edges, g, start_time, deadline):
                return n, edges

    # ---- 3. Voltage-lift construction (guaranteed high girth, fast) ----
    lifted = _voltage_lift(k, g, start_time, deadline)
    if lifted is not None:
        n, edges = lifted
        if best is None or n < best[0]:
            best = (n, edges)

    # ---- 4. LUW fallback ----
    n, edges = _luw_fallback(k, g)
    if best is None or n < best[0]:
        best = (n, edges)
    return best


def _check_girth_circulant(n, a, g, start_time, deadline):
    """Non-backtracking walk check for C(n; {a, -a, n/2}) using NumPy."""
    h = n // 2
    S = [a % n, (n - a) % n, h % n]
    neg_idx = [1, 0, 2]

    R = np.zeros((3, n), dtype=bool)
    R[0, S[0]] = True
    R[1, S[1]] = True
    R[2, S[2]] = True

    for d in range(2, g):
        if time.time() - start_time > deadline:
            return False
        R_new = np.zeros((3, n), dtype=bool)
        for i in range(3):
            for j in range(3):
                if j == neg_idx[i]:
                    continue
                R_new[j] |= np.roll(R[i], S[j])
        R = R_new
        if R[:, 0].any():
            return False

    return True


def _generate_random_3_reg(n):
    """Random simple 3-regular graph on n vertices (configuration model)."""
    if n % 2 != 0 or n < 4:
        return None
    for _ in range(100):
        stubs = [i for i in range(n) for _ in range(3)]
        random.shuffle(stubs)
        edges = []
        valid = True
        edge_set = set()
        for i in range(0, 3 * n, 2):
            u = stubs[i]
            v = stubs[i + 1]
            if u == v:
                valid = False
                break
            if u > v:
                u, v = v, u
            if (u, v) in edge_set:
                valid = False
                break
            edge_set.add((u, v))
            edges.append((u, v))
        if valid:
            return edges
    return None


def _is_connected(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    visited = [False] * n
    q = deque([0])
    visited[0] = True
    count = 1
    while q:
        u = q.popleft()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                count += 1
                q.append(v)
    return count == n


def _check_girth_general(n, edges, g, start_time, deadline):
    """BFS from every vertex, detect any cycle of length < g."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    for start in range(n):
        if time.time() - start_time > deadline:
            return False
        visited = {start: 0}
        q = deque([(start, -1, 0)])
        while q:
            u, parent, d = q.popleft()
            for v in adj[u]:
                if v == parent:
                    continue
                if v in visited:
                    dv = visited[v]
                    if d + dv + 1 < g:
                        return False
                else:
                    visited[v] = d
                    q.append((v, u, d + 1))
    return True


def _voltage_lift(k, g, start_time, deadline):
    """
    Voltage-graph (lift) construction: a k-regular base graph B with edges
    labeled by elements of a finite abelian group G (order m). The cover has
    |V(B)|*m vertices and is k-regular. If the base girth is >= g, the cover
    girth is >= g (non-backtracking walks project to non-backtracking walks
    on the base). We pick a random k-regular base of girth >= g and a prime
    group order m so the cover has as few vertices as possible.
    """
    if _HAS_NUMPY:
        # Vectorized base girth check
        def base_girth_ok(adj, m, g):
            nb = np.zeros((k, m), dtype=bool)
            for i, (u, v, l) in enumerate(adj):
                nb[i, l] = True
            for _ in range(2, g):
                if time.time() - start_time > deadline:
                    return False
                nb_new = np.zeros((k, m), dtype=bool)
                for i in range(k):
                    for j in range(k):
                        if j == k - 1 - i:
                            continue
                        nb_new[j] |= np.roll(nb[i], adj[j][2])
                nb = nb_new
                if nb[:, 0].any():
                    return False
            return True
    else:
        def base_girth_ok(adj, m, g):
            nb = [set() for _ in range(k)]
            for i, (u, v, l) in enumerate(adj):
                nb[i].add(l)
            for _ in range(2, g):
                if time.time() - start_time > deadline:
                    return False
                nb_new = [set() for _ in range(k)]
                for i in range(k):
                    for j in range(k):
                        if j == k - 1 - i:
                            continue
                        for x in nb[i]:
                            nb_new[j].add((x + adj[j][2]) % m)
                nb = nb_new
                for s in nb:
                    if 0 in s:
                        return False
            return True

    # Candidate group orders (primes give clean, connected covers).
    # Cover size = |V(B)| * m; keep m small.
    group_orders = [3, 5, 7, 11, 13, 17, 19, 23]
    # Base graph sizes: even, k-regular, need girth >= g.
    base_sizes = [16, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72, 80]

    best = None
    for m in group_orders:
        if time.time() - start_time > deadline:
            break
        for nb in base_sizes:
            if time.time() - start_time > deadline:
                break
            if nb * m > 936:
                continue  # no better than known best; skip
            if nb < 2 * g:
                continue
            # Build a random k-regular base graph with girth >= g.
            for _ in range(8):
                if time.time() - start_time > deadline:
                    break
                badges = _generate_random_3_reg(nb)
                if badges is None:
                    continue
                if not _is_connected(nb, badges):
                    continue
                if not _check_girth_general(nb, badges, g, start_time, deadline):
                    continue
                # Assign random distinct labels to edges.
                labels = list(range(nb * k // 2))
                random.shuffle(labels)
                adj = []
                for (u, v), l in zip(badges, labels):
                    adj.append((u, v, l % m))
                if not base_girth_ok(adj, m, g):
                    continue
                # Build the cover.
                N = nb * m
                cover_edges = []
                seen = set()
                for (u, v, l) in adj:
                    for g0 in range(m):
                        a = u * m + g0
                        b = v * m + (g0 + l) % m
                        if a == b:
                            continue
                        if a > b:
                            a, b = b, a
                        if (a, b) in seen:
                            continue
                        seen.add((a, b))
                        cover_edges.append((a, b))
                if best is None or N < best[0]:
                    best = (N, cover_edges)
    return best


def _luw_fallback(k, g):
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
