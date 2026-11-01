"""Fresh design: warm-start from LUW graph, then reduce vertex count via
valid edge swaps that preserve girth. Girth certified by non-backtracking BFS."""
import time
import random
from collections import deque

_DEADLINE = 150.0
_MARGIN = 12.0  # safety margin before the 150s kill


def _luw(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q): q-regular bipartite, girth >= d+5 (odd d)."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
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


def _girth_ok(adj, n, g):
    """Return True iff no cycle shorter than g. Non-backtracking BFS from every vertex."""
    # For each start vertex, do a BFS over (node, prev) states, tracking depth.
    # A cycle of length L < g is detected if we reach a state (node, prev) at depth
    # >= L where the node was already seen at depth <= L - 3 (closing a non-backtracking walk).
    # Simpler exact certificate: enumerate non-backtracking walks up to depth g-1;
    # if any walk of length L (L < g) is closed (returns to start without backtracking), girth violated.
    # Efficient version: BFS from each start over directed edges (u->v, v!=prev), depth up to g-1.
    # Mark visited[start][v] = first depth. A closed non-backtracking walk of length L < g
    # appears as reaching (start, last) with last != prev-of-start and depth L.
    for s in range(n):
        # state: (node, prev), depth
        seen = {}  # (node, prev) -> depth
        q = deque()
        # initial: from s to each neighbor, depth 1
        for v in adj[s]:
            q.append((v, s, 1))
            seen[(v, s)] = 1
        while q:
            node, prev, depth = q.popleft()
            if depth >= g - 1:
                continue
            for v in adj[node]:
                if v == prev:
                    continue
                # closing a cycle: v == s and depth+1 < g and s != prev (non-backtracking close)
                if v == s:
                    if depth + 1 < g:
                        return False
                    continue
                st = (v, node)
                if st in seen:
                    continue
                seen[st] = depth + 1
                q.append((v, node, depth + 1))
    return True


def _build_adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def construct(k, g):
    t0 = time.time()
    rng = random.Random(12345)

    # Warm start: LUW graph (valid, girth >= g)
    n, edges = _luw(k, g)
    adj = _build_adj(n, edges)

    # Quick validity check of warm start
    if not _girth_ok(adj, n, g):
        # Should not happen; fallback to pure return
        return n, edges

    best_n = n
    best_edges = list(edges)
    best_adj = [list(a) for a in adj]

    # Local search: try to reduce vertex count is hard directly (regular graph vertex count
    # is fixed by construction). Instead we can't easily reduce n via edge swaps (n is fixed).
    # So the real lever: try alternative constructions / smaller n candidates and repair.
    # For 3-regular, n must be even. Try to find a valid graph on fewer vertices via
    # random generation + repair, using the deadline.

    deadline = t0 + _DEADLINE - _MARGIN

    # Moore bound lower bound on n for k=3, girth g
    if g % 2 == 0:
        # Moore bound: n >= 2 * sum_{i=0}^{g/2 - 1} (k-1)^i
        mb = 2
        term = 1
        for i in range(g // 2 - 1):
            mb += term
            term *= (k - 1)
    else:
        # n >= (k*(k-1)^{g/2} - k) / (k-1)  (approx)
        mb = 0
        term = k
        for i in range(g // 2):
            mb += term
            term *= (k - 1)
    mb = max(mb, k)
    if mb % 2:
        mb += 1

    # Try candidate n values from mb upward (even), generate random 3-regular, repair girth.
    # Budget: try a handful of n's, each with a few random starts.
    n_candidates = []
    nn = mb
    while nn < best_n and len(n_candidates) < 40:
        n_candidates.append(nn)
        nn += 2

    def gen_random_3reg(n):
        """Generate a random 3-regular simple graph on n vertices (n even, n>=4)."""
        if n < 4 or n % 2:
            return None
        # Configuration model with perfect matching, then de-loop/de-multi
        half = n * 3 // 2
        stubs = []
        for v in range(n):
            stubs += [v] * 3
        for _ in range(60):
            rng.shuffle(stubs)
            pairs = [(stubs[i], stubs[i + 1]) for i in range(0, half, 2)]
            # check no loops and no multiedges
            seen = set()
            ok = True
            for a, b in pairs:
                if a == b:
                    ok = False
                    break
                key = (min(a, b), max(a, b))
                if key in seen:
                    ok = False
                    break
                seen.add(key)
            if ok:
                return pairs
        return None

    def repair_girth(adj, n, g, time_limit):
        """Try edge swaps to eliminate cycles shorter than g. Returns (ok, adj)."""
        adj = [list(a) for a in adj]
        end = time.time() + time_limit
        while time.time() < end:
            # find a short cycle
            short = _find_short_cycle(adj, n, g)
            if short is None:
                return True, adj
            # short is list of vertices forming a cycle of length L < g
            L = len(short)
            # break the cycle by swapping one edge on the cycle with an off-cycle edge
            # pick a random edge (u,v) on the cycle, find a partner edge (x,y) off cycle
            # such that swap keeps 3-regular and simple and doesn't create shorter cycle.
            for _ in range(200):
                i = rng.randrange(L)
                u = short[i]
                v = short[(i + 1) % L]
                # find x in adj[u] \ {v, short[i-1]} and y in adj[v] \ {u, short[i+1]}
                cand_u = [w for w in adj[u] if w != v and w != short[i - 1]]
                cand_v = [w for w in adj[v] if w != u and w != short[(i + 1) % L]]
                if not cand_u or not cand_v:
                    continue
                x = rng.choice(cand_u)
                y = rng.choice(cand_v)
                if x == y:
                    continue
                # ensure (u,y) and (v,x) not already edges
                if y in adj[u] or x in adj[v]:
                    continue
                # perform swap: remove (u,v) and (x,y); add (u,y) and (v,x)
                # but (x,y) must be an existing edge. x in adj[u] means edge (u,x) exists.
                # We need an existing edge (x, y) where x in adj[u], y in adj[v], x!=v, y!=u.
                # Redo: pick x in adj[u]\{v}, y in adj[v]\{u} with edge (x,y) present.
                found = False
                for x in cand_u:
                    if x == v:
                        continue
                    for y in cand_v:
                        if y == u:
                            continue
                        if x == y:
                            continue
                        if y in adj[x] and x in adj[y]:
                            # candidate swap: (u,v)+(x,y) -> (u,y)+(v,x)
                            if y in adj[u] or x in adj[v]:
                                continue
                            # check new edges not duplicate
                            # apply
                            adj[u].remove(v); adj[v].remove(u)
                            adj[x].remove(y); adj[y].remove(x)
                            adj[u].append(y); adj[y].append(u)
                            adj[v].append(x); adj[x].append(v)
                            if _girth_ok(adj, n, g):
                                found = True
                                break
                            else:
                                # revert
                                adj[u].remove(y); adj[y].remove(u)
                                adj[v].remove(x); adj[x].remove(v)
                                adj[u].append(v); adj[v].append(u)
                                adj[x].append(y); adj[y].append(x)
                    if found:
                        break
                if found:
                    break
            if not found:
                # try a random swap elsewhere
                pass
        return _girth_ok(adj, n, g), adj

    def _find_short_cycle(adj, n, g):
        for s in range(n):
            seen = {}
            q = deque()
            for v in adj[s]:
                q.append((v, s, 1, [s, v]))
                seen[(v, s)] = 1
            while q:
                node, prev, depth, path = q.popleft()
                if depth >= g - 1:
                    continue
                for v in adj[node]:
                    if v == prev:
                        continue
                    if v == s:
                        if depth + 1 < g:
                            return path + [s]
                        continue
                    st = (v, node)
                    if st in seen:
                        continue
                    seen[st] = depth + 1
                    q.append((v, node, depth + 1, path + [v]))
        return None

    for nn in n_candidates:
        if time.time() > deadline:
            break
        for attempt in range(5):
            if time.time() > deadline:
                break
            pairs = gen_random_3reg(nn)
            if pairs is None:
                continue
            adj = _build_adj(nn, pairs)
            if not _girth_ok(adj, nn, g):
                ok, adj = repair_girth(adj, nn, g, time_limit=max(1.0, deadline - time.time() - 0.5))
            if ok and _girth_ok(adj, nn, g):
                # valid smaller graph
                best_n = nn
                best_edges = []
                for u in range(nn):
                    for v in adj[u]:
                        if u < v:
                            best_edges.append((u, v))
                if nn <= mb:
                    break
        if best_n <= mb:
            break

    return best_n, best_edges
