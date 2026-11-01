"""Rewritten program: randomized 4-regular construction + girth-repair via 2-switches.
A completely different approach family from explicit algebraic (LUW) cages."""
import random
import sys

# EVOLVE-BLOCK-START

def _has_short_cycle(adj, g, max_check=4):
    """Return a cycle of length < g found by non-backtracking walks, or None.
    max_check limits how many BFS roots we probe for speed."""
    n = len(adj)
    # non-backtracking walk from each start, track (prev, cur) states
    for s in range(n):
        # state: (prev, cur) -> (start, depth)
        from collections import deque
        seen = {}
        dq = deque()
        for nb in adj[s]:
            state = (s, nb)
            seen[state] = (s, 1, None)  # (start, depth, path_prev)
            dq.append(state)
        while dq:
            prev, cur = dq.popleft()
            st = seen[(prev, cur)]
            start, depth, _ = st
            if depth >= g:
                continue
            for nb in adj[cur]:
                if nb == prev:
                    continue
                nstate = (cur, nb)
                if nb == start and depth + 1 < g:
                    # found a cycle of length depth+1
                    return depth + 1
                if nstate not in seen:
                    seen[nstate] = (start, depth + 1, prev)
                    dq.append(nstate)
    return None


def _find_short_cycle_path(adj, g):
    """Return an actual cycle (list of vertices) of length < g, or None."""
    n = len(adj)
    from collections import deque
    for s in range(n):
        # BFS in the non-backtracking state graph, storing parent to reconstruct
        parent = {}
        dq = deque()
        for nb in adj[s]:
            state = (s, nb)
            parent[state] = None
            dq.append(state)
        while dq:
            prev, cur = dq.popleft()
            depth = 0
            # reconstruct depth by walking parent
            d = 1
            st = (prev, cur)
            while parent.get(st) is not None:
                st = parent[st]
                d += 1
            for nb in adj[cur]:
                if nb == prev:
                    continue
                if nb == s:
                    if d + 1 < g:
                        # reconstruct cycle
                        cycle = [s, cur]
                        st = (prev, cur)
                        while parent.get(st) is not None:
                            p, c = parent[st]
                            cycle.append(p)
                            st = (p, c)
                        cycle.reverse()
                        # cycle: s ... cur, and cur-s closes it
                        return cycle
                nstate = (cur, nb)
                if nstate not in parent:
                    parent[nstate] = (prev, cur)
                    dq.append(nstate)
    return None


def _random_4_regular(n, rng):
    """Build a random simple 4-regular graph on n vertices via configuration model + fix."""
    # configuration model: 4 stubs per vertex, random pairing
    while True:
        stubs = []
        for v in range(n):
            stubs.extend([v] * 4)
        rng.shuffle(stubs)
        adj = {v: [] for v in range(n)}
        ok = True
        for i in range(0, len(stubs), 2):
            a, b = stubs[i], stubs[i + 1]
            if a == b:
                ok = False
                break
            if b in adj[a]:
                ok = False
                break
            adj[a].append(b)
            adj[b].append(a)
        if ok:
            return adj
    return None


def _improve_girth(adj, g, rng, max_iters=20000):
    """2-switch repair: find short cycle, swap an edge to break it, keep 4-regular & simple."""
    n = len(adj)
    adj_set = {v: set(adj[v]) for v in range(n)}
    for it in range(max_iters):
        cyc = _find_short_cycle_path(adj, g)
        if cyc is None:
            return adj
        # break the cycle: pick two non-adjacent-in-cycle edges (a,b) and (c,d) and swap to (a,c),(b,d)
        m = len(cyc)
        if m < 4:
            continue
        # try several random pairs of opposite edges
        for _ in range(30):
            i = rng.randrange(m)
            j = rng.randrange(m)
            if (i - j) % m in (0, 1, m - 1):
                continue
            a, b = cyc[i], cyc[(i + 1) % m]
            c, d = cyc[j], cyc[(j + 1) % m]
            # ensure a,b and c,d are actual edges and not overlapping
            if b not in adj_set[a] or d not in adj_set[c]:
                continue
            if len({a, b, c, d}) < 4:
                continue
            # candidate new edges (a,c) and (b,d) must not already exist
            if c in adj_set[a] or d in adj_set[b]:
                continue
            # apply
            adj_set[a].discard(b); adj_set[b].discard(a)
            adj_set[c].discard(d); adj_set[d].discard(c)
            adj_set[a].add(c); adj_set[c].add(a)
            adj_set[b].add(d); adj_set[d].add(b)
            break
    return adj


def _verify(adj, k, g):
    n = len(adj)
    if n < k + 1:
        return False
    for v in range(n):
        if len(adj[v]) != k:
            return False
        if len(set(adj[v])) != k:
            return False
    # simple
    for v in range(n):
        for u in adj[v]:
            if u == v:
                return False
            if v not in adj[u]:
                return False
    # girth
    if _has_short_cycle(adj, g) is not None:
        return False
    return True


def construct(k, g):
    if k != 4:
        # fallback: use a simple construction for general k if needed
        # For this task k=4; provide a basic fallback via LUW-like for safety.
        return _fallback(k, g)

    rng = random.Random(12345)
    # Moore bound for (4,10): 1 + 4*(1+4+16+64) = 341; known cage 320.
    # Try to build a 4-regular graph with girth >= 10 on as few vertices as possible.
    # Start from a candidate n and try random constructions + repair.
    best = None
    # candidate sizes to try (lower is better); we attempt to find valid graphs
    candidates = [320, 324, 328, 332, 336, 340, 344, 348, 352, 360, 380, 400]
    for n in candidates:
        if n < 1 + k * (1 + k + k**2 + k**3):  # below Moore bound, impossible
            continue
        for attempt in range(6):
            adj = _random_4_regular(n, rng)
            if adj is None:
                continue
            adj = _improve_girth(adj, g, rng, max_iters=4000)
            if _verify(adj, k, g):
                edges = []
                for v in range(n):
                    for u in adj[v]:
                        if v < u:
                            edges.append((v, u))
                if best is None or n < best[0]:
                    best = (n, edges)
                # found a valid one at this n; since we go low->high, this is minimal found
                return n, edges
    if best is not None:
        return best
    # last resort fallback
    return _fallback(k, g)


def _fallback(k, g):
    # Reuse a simple explicit construction (LUW) as a guaranteed-valid fallback.
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

# EVOLVE-BLOCK-END
