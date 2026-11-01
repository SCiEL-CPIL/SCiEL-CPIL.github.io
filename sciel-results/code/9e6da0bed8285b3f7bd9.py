"""Cage via shortest-cycle-breaking local search + LUW fallback. New approach family."""
import time
from collections import deque

_T0 = time.time()
_LIMIT = 105.0


def _edges(adj):
    n = len(adj)
    e = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                e.append((i, j))
    return e


def _girth_ok(adj, g):
    n = len(adj)
    for s in range(n):
        seen = {}
        dq = deque()
        for nb in adj[s]:
            seen[(nb, s)] = 1
            dq.append((nb, s))
        while dq:
            (v, p), d = dq.popleft()
            if d >= g - 1:
                continue
            for w in adj[v]:
                if w == p:
                    continue
                if w == s:
                    return False
                st = (w, v)
                if st not in seen:
                    seen[st] = d + 1
                    dq.append(st)
    return True


def _shortest_cycle(adj):
    n = len(adj)
    best = None
    for s in range(n):
        # BFS with parent to reconstruct a cycle
        parent = {}
        dist = {}
        dq = deque()
        for nb in adj[s]:
            parent[(nb, s)] = None
            dist[(nb, s)] = 1
            dq.append((nb, s))
        found = None
        while dq:
            (v, p), d = dq.popleft()
            if d >= 10:  # we only care up to g-1 ~ 9; cap for speed
                continue
            for w in adj[v]:
                if w == p:
                    continue
                if w == s:
                    # cycle length d+1
                    if found is None or d + 1 < found[0]:
                        # reconstruct
                        cyc = [w]
                        cur = (v, p)
                        while cur is not None:
                            a, b = cur
                            cyc.append(a)
                            cur = parent.get((a, b))
                        cyc.append(s)
                        found = (d + 1, cyc)
                    continue
                st = (w, v)
                if st not in dist:
                    dist[st] = d + 1
                    parent[st] = (v, p)
                    dq.append(st)
        if found is not None:
            if best is None or found[0] < best[0]:
                best = found
            if best[0] <= 3:
                break
    return best


def _rand_kreg(n, k, seed):
    import random
    rng = random.Random(seed)
    for _ in range(300):
        half = []
        for v in range(n):
            half.extend([v] * k)
        rng.shuffle(half)
        adj = [set() for _ in range(n)]
        ok = True
        for i in range(0, len(half), 2):
            a, b = half[i], half[i + 1]
            if a == b or b in adj[a]:
                ok = False
                break
            adj[a].add(b)
            adj[b].add(a)
        if ok and all(len(adj[v]) == k for v in range(n)):
            return adj
    return None


def _repair(adj, g, seed):
    import random
    rng = random.Random(seed)
    n = len(adj)
    for it in range(60000):
        if time.time() - _T0 > _LIMIT:
            return None
        if _girth_ok(adj, g):
            return adj
        cyc = _shortest_cycle(adj)
        if cyc is None:
            return adj
        L, nodes = cyc
        if L >= g:
            return adj
        # Break the cycle: pick an edge (a,b) on the cycle and swap with (c,d)
        # where c,d are not on the cycle and not adjacent to a or b, to avoid
        # creating a shorter cycle.
        # nodes is a list forming the cycle (with possible duplicates at ends).
        # Extract unique cycle edges.
        cyc_edges = []
        seen_set = set()
        for i in range(len(nodes) - 1):
            a, b = nodes[i], nodes[i + 1]
            if a == b:
                continue
            key = (min(a, b), max(a, b))
            if key in seen_set:
                continue
            seen_set.add(key)
            cyc_edges.append((a, b))
        if not cyc_edges:
            continue
        a, b = rng.choice(cyc_edges)
        # find c,d
        for _ in range(50):
            c = rng.randrange(n)
            d = rng.randrange(n)
            if c == d or c == a or c == b or d == a or d == b:
                continue
            if c in adj[a] or d in adj[b] or c in adj[d]:
                continue
            # perform swap
            adj[a].discard(b); adj[b].discard(a)
            adj[a].add(c); adj[c].add(a)
            adj[b].add(d); adj[d].add(b)
            # accept if no new cycle shorter than L
            if _girth_ok(adj, L):
                break
            else:
                # revert
                adj[a].discard(c); adj[c].discard(a)
                adj[b].discard(d); adj[d].discard(b)
                adj[a].add(b); adj[b].add(a)
    return None


def _luw(k, g):
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


def construct(k, g):
    if k == 2:
        return g, [(i, (i + 1) % g) for i in range(g)]

    if g % 2 == 0:
        moore = k * (k - 1) ** (g // 2 - 1)
    else:
        moore = 1 + k * sum((k - 1) ** i for i in range((g - 3) // 2 + 1))

    luw_n, luw_e = _luw(k, g)

    best_n, best_e = luw_n, luw_e

    for n in range(moore, min(moore + 120, luw_n)):
        if (k * n) % 2 != 0:
            continue
        if time.time() - _T0 > _LIMIT:
            break
        adj = _rand_kreg(n, k, seed=n * 131 + 7)
        if adj is None:
            continue
        fixed = _repair(adj, g, seed=n * 17 + 3)
        if fixed is not None and _girth_ok(fixed, g):
            best_n, best_e = n, _edges(fixed)
            break  # smallest n found

    return best_n, best_e
