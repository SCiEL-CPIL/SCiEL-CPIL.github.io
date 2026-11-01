"""Small 3-regular graphs of large girth.

Combines a cycle-avoiding greedy local search (2-switches that break the
shortest cycle) with structured voltage-lift starts and the LUW construction
as a guaranteed fallback.  The local search drives the girth of a fixed-size
3-regular graph up to the target, which is a different modelling family than
the pure random-search / lift approach of the seed.
"""
import time
import random
import numpy as np
from collections import deque


# ---------------------------------------------------------------------------
# Girth checking (vectorized numpy, returns true girth)
# ---------------------------------------------------------------------------
def _girth_np(adj, n, g):
    """Return the girth of a 3-regular graph (adj is an (n,3) int array).
    Returns >= g if no cycle shorter than g exists.  Fast: per start we
    BFS non-backtracking states and stop as soon as a cycle is found or the
    depth reaches g-1."""
    for start in range(n):
        # state index: (v * n + prev) ; prev == -1 -> v*n + n
        dist = np.full(n * (n + 1), -1, dtype=np.int32)
        start_state = start * (n + 1) + n
        dist[start_state] = 0
        dq = deque([start_state])
        while dq:
            s = dq.popleft()
            d = dist[s]
            if d >= g - 1:
                continue
            v = s // (n + 1)
            prev = s % (n + 1)
            if prev == n:
                prev = -1
            for w in adj[v]:
                if w == prev:
                    continue
                if w == start and d + 1 >= 3:
                    return d + 1
                ns = w * (n + 1) + v
                if dist[ns] < 0:
                    dist[ns] = d + 1
                    dq.append(ns)
    return g


def _verify_girth(adj, n, g):
    """Legacy list-based girth check (kept for compatibility)."""
    for start in range(n):
        dist = {}
        dq = deque([(start, -1)])
        dist[(start, -1)] = 0
        while dq:
            v, prev = dq.popleft()
            d = dist[(v, prev)]
            if d >= g - 1:
                continue
            for w in adj[v]:
                if w == prev:
                    continue
                if w == start and d + 1 >= 3:
                    return False
                state = (w, v)
                if state not in dist:
                    dist[state] = d + 1
                    dq.append(state)
    return True


# ---------------------------------------------------------------------------
# 2-switch local search: break the shortest cycle, preserve 3-regularity
# ---------------------------------------------------------------------------
def _find_shortest_cycle(adj, n, g):
    """Return a list of vertices forming the shortest cycle (length < g),
    or None.  Uses BFS from each vertex; the cycle is the first re-encounter
    of a vertex at depth >= 2."""
    for start in range(n):
        parent = [-1] * n
        depth = [-1] * n
        depth[start] = 0
        dq = deque([start])
        while dq:
            v = dq.popleft()
            d = depth[v]
            if d >= g - 1:
                continue
            for w in adj[v]:
                if depth[w] < 0:
                    depth[w] = d + 1
                    parent[w] = v
                    dq.append(w)
                elif w != parent[v] and depth[w] <= d:
                    # found a cycle: start .. v - w .. start
                    cycle = [v]
                    cur = v
                    while cur != start:
                        cur = parent[cur]
                        cycle.append(cur)
                    # prepend the w-chain
                    tail = [w]
                    cur = w
                    while cur != start:
                        cur = parent[cur]
                        tail.append(cur)
                    tail.reverse()
                    full = cycle + tail[:-1]
                    # dedupe consecutive duplicates
                    cleaned = [full[0]]
                    for x in full[1:]:
                        if x != cleaned[-1]:
                            cleaned.append(x)
                    if len(cleaned) >= 3:
                        return cleaned
    return None


def _local_search(adj, n, g, deadline, max_iter=400):
    """Repeatedly find the shortest cycle and apply a 2-switch to break it.
    Returns True once girth >= g, else False.  Mutates adj in place."""
    cur = _girth_np(adj, n, g)
    if cur >= g:
        return True
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        cyc = _find_shortest_cycle(adj, n, g)
        if cyc is None:
            return True
        # pick two non-adjacent edges of the cycle to swap
        L = len(cyc)
        cand = []
        for i in range(L):
            a = cyc[i]
            b = cyc[(i + 1) % L]
            for j in range(i + 2, L):
                if j == (i + L - 1) % L:
                    continue
                c = cyc[j]
                d = cyc[(j + 1) % L]
                if c == a or c == b or d == a or d == d:
                    continue
                if c == b or d == a:
                    continue
                cand.append((a, b, c, d))
        if not cand:
            # only tiny cycle; do a random 2-switch to escape
            cand = None
        if cand is None:
            edges = []
            for v in range(n):
                for w in adj[v]:
                    if v < w:
                        edges.append((v, w))
            a, b = random.choice(edges)
            others = [e for e in edges if e != (a, b)]
            if not others:
                return _girth_np(adj, n, g) >= g
            c, d = random.choice(others)
        else:
            a, b, c, d = random.choice(cand)
        # apply swap: remove (a,b),(c,d); add (a,c),(b,d)
        adj[a] = list(x for x in adj[a] if x != b) + [c]
        adj[b] = list(x for x in adj[b] if x != a) + [d]
        adj[c] = list(x for x in adj[c] if x != d) + [a]
        adj[d] = list(x for x in adj[d] if x != c) + [b]
        newg = _girth_np(adj, n, g)
        if newg > cur:
            cur = newg
            if cur >= g:
                return True
        else:
            # revert (keep the better state)
            adj[a] = list(x for x in adj[a] if x != c) + [b]
            adj[b] = list(x for x in adj[b] if x != d) + [a]
            adj[c] = list(x for x in adj[c] if x != a) + [d]
            adj[d] = list(x for x in adj[d] if x != b) + [c]
    return _girth_np(adj, n, g) >= g


# ---------------------------------------------------------------------------
# Voltage lifts
# ---------------------------------------------------------------------------
def _build_lift(base_edges, m, shifts):
    n_base = max(max(u for u, v in base_edges), max(v for u, v in base_edges)) + 1
    n = n_base * m
    adj = [set() for _ in range(n)]
    for idx, (u, v) in enumerate(base_edges):
        vol = shifts[idx] % m
        for i in range(m):
            a = u * m + i
            b = v * m + (i + vol) % m
            adj[a].add(b)
            adj[b].add(a)
    for i in range(n):
        if len(adj[i]) != 3:
            return None
    return n, np.array([sorted(adj[i]) for i in range(n)], dtype=np.int64)


_PETERSEN = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
             (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
             (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)]
_K33 = [(0, 3), (0, 4), (0, 5), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5)]
_DODECA = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
           (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),
           (5, 6), (6, 7), (7, 8), (8, 9), (9, 5),
           (5, 10), (6, 11), (7, 12), (8, 13), (9, 14),
           (10, 11), (11, 12), (12, 13), (13, 14), (14, 10),
           (10, 15), (11, 16), (12, 17), (13, 18), (14, 19),
           (15, 16), (16, 17), (17, 18), (18, 19), (19, 15)]


# ---------------------------------------------------------------------------
# LUW construction (guaranteed fallback)
# ---------------------------------------------------------------------------
def _luw_construct(k, g, d_override=None):
    q = k
    if d_override is not None:
        d = d_override
    else:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), \
                        (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), \
                        (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
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
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 \
                    else sub(x[c], mul(x[a], y[b]))
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


def _random_3reg(n, rng):
    """Return an (n,3) array for a random 3-regular graph, or None."""
    if n % 2 != 0:
        return None
    for _ in range(200):
        adj = [set() for _ in range(n)]
        ok = True
        for i in range(3):
            perm = list(range(n))
            rng.shuffle(perm)
            for j in range(0, n, 2):
                a, b = perm[j], perm[j + 1]
                if a == b or b in adj[a]:
                    ok = False
                    break
                adj[a].add(b)
                adj[b].add(a)
            if not ok:
                break
        if ok and all(len(adj[v]) == 3 for v in range(n)):
            return np.array([sorted(adj[v]) for v in range(n)], dtype=np.int64)
    return None


def construct(k, g):
    start_time = time.time()
    deadline = start_time + 128  # safety margin before 150s limit

    if k != 3:
        n_luw, edges_luw = _luw_construct(k, g)
        return n_luw, edges_luw

    if g % 2 == 0:
        moore = 2 * (2 ** (g // 2) - 1)
    else:
        moore = 2 * (2 ** (g // 2 - 1) - 1) + 2

    def finish(adj, n):
        edges = []
        for v in range(n):
            for w in adj[v]:
                if v < w:
                    edges.append((v, int(w)))
        return n, edges

    # --- Phase 1: structured voltage-lift starts + local search -------------
    base_graphs = [
        ("petersen", _PETERSEN, 10),
        ("k33", _K33, 6),
        ("dodeca", _DODECA, 20),
    ]
    for name, base_edges, n_base in base_graphs:
        if time.time() > deadline - 20:
            break
        e_count = len(base_edges)
        n_min = max(1, (moore + n_base - 1) // n_base)
        for m in range(n_min, n_min + 120):
            if time.time() > deadline - 15:
                break
            n = n_base * m
            if n < moore:
                continue
            for attempt in range(3):
                if time.time() > deadline - 12:
                    break
                rng = random.Random(hash((name, m, attempt)))
                shifts = [rng.randint(0, m - 1) for _ in range(e_count)]
                res = _build_lift(base_edges, m, shifts)
                if res is None:
                    continue
                n_r, adj = res
                if _girth_np(adj, n_r, g) >= g:
                    return finish(adj, n_r)
                # local search to push girth up
                if _local_search(adj, n_r, g, deadline - 12):
                    return finish(adj, n_r)

    # --- Phase 2: pure-random starts at the Moore bound + local search ------
    if time.time() < deadline - 25:
        for n_try in range(moore, moore + 40, 2):
            if time.time() > deadline - 20:
                break
            for attempt in range(4):
                if time.time() > deadline - 18:
                    break
                rng = random.Random(hash(("rand", n_try, attempt)))
                adj = _random_3reg(n_try, rng)
                if adj is None:
                    continue
                if _girth_np(adj, n_try, g) >= g:
                    return finish(adj, n_try)
                if _local_search(adj, n_try, g, deadline - 18):
                    return finish(adj, n_try)

    # --- Phase 3: LUW fallback (guaranteed valid) ---------------------------
    for d_try in [7, 8, 9, 10, 11, 12, 13, 14]:
        if time.time() > deadline - 3:
            break
        try:
            n_try, edges_try = _luw_construct(3, g, d_override=d_try)
            adj = [set() for _ in range(n_try)]
            for u, v in edges_try:
                adj[u].add(v)
                adj[v].add(u)
            adj_arr = np.array([sorted(adj[v]) for v in range(n_try)], dtype=np.int64)
            if _girth_np(adj_arr, n_try, g) >= g:
                return n_try, edges_try
        except Exception:
            pass

    n_luw, edges_luw = _luw_construct(3, g)
    return n_luw, edges_luw
