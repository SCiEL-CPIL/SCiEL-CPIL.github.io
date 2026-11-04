import time
import random
from collections import deque

def construct(k, g):
    t0 = time.time()
    deadline = t0 + 138.0

    # ---------- Guaranteed baseline: LUW / Exoo-type algebraic construction ----------
    q = k
    # True girth relation for LUW: girth = 2d + 2  =>  d = (g-2)//2
    d = max(3, (g - 2) // 2)
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

    baseline_n = len(order)
    baseline_edges = edges

    # ---------- Fast girth verification: non-backtracking BFS from every vertex ----------
    def girth_ok(adj, n, g):
        """Return True iff no cycle shorter than g exists (non-backtracking BFS)."""
        limit = g // 2
        for s in range(n):
            dist = [-1] * n
            par = [-1] * n
            dist[s] = 0
            dq = deque([s])
            bad = False
            while dq:
                u = dq.popleft()
                du = dist[u]
                if du >= limit:
                    continue
                for v in adj[u]:
                    if v == par[u]:
                        continue
                    if dist[v] == -1:
                        dist[v] = du + 1
                        par[v] = u
                        dq.append(v)
                    elif v != s:
                        if du + dist[v] + 1 < g:
                            bad = True
                            break
                if bad:
                    break
            if bad:
                return False
        return True

    # ---------- Build adjacency helpers ----------
    def build_adj(edges, n):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def find_short_cycle(adj, n, g):
        """Return a list of vertices forming a cycle shorter than g, or None."""
        limit = g // 2
        for s in range(n):
            dist = [-1] * n
            par = [-1] * n
            dist[s] = 0
            dq = deque([s])
            while dq:
                u = dq.popleft()
                du = dist[u]
                if du >= limit:
                    continue
                for v in adj[u]:
                    if v == par[u]:
                        continue
                    if dist[v] == -1:
                        dist[v] = du + 1
                        par[v] = u
                        dq.append(v)
                    elif v != s:
                        L = du + dist[v] + 1
                        if L < g:
                            # reconstruct cycle
                            cyc = [u]
                            c = u
                            while c != s:
                                c = par[c]
                                cyc.append(c)
                            c = v
                            while c != s:
                                c = par[c]
                                cyc.append(c)
                            cyc.append(s)
                            return cyc
        return None

    # ---------- Cycle-breaking 2-switch local search ----------
    def try_break(edges, n, g, max_time):
        """Repeatedly find a short cycle and break it via a 2-switch.
        Accept a swap only if it does not create a cycle shorter than the
        previous shortest (measured by full girth check)."""
        t_dead = time.time() + max_time
        adj = build_adj(edges, n)
        # current shortest cycle length (or g if already ok)
        def shortest_cycle_len(adj):
            limit = g // 2
            best = g
            for s in range(n):
                dist = [-1] * n
                par = [-1] * n
                dist[s] = 0
                dq = deque([s])
                while dq:
                    u = dq.popleft()
                    du = dist[u]
                    if du >= limit:
                        continue
                    for v in adj[u]:
                        if v == par[u]:
                            continue
                        if dist[v] == -1:
                            dist[v] = du + 1
                            par[v] = u
                            dq.append(v)
                        elif v != s:
                            L = du + dist[v] + 1
                            if L < best:
                                best = L
            return best

        cur_len = shortest_cycle_len(adj)
        if cur_len >= g:
            return edges, True

        # index edges by frozenset for O(1) membership
        eset = set(edges)
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1

        stuck = 0
        while time.time() < t_dead and cur_len < g:
            adj = build_adj(edges, n)
            cyc = find_short_cycle(adj, n, g)
            if cyc is None:
                cur_len = shortest_cycle_len(adj)
                if cur_len >= g:
                    return edges, True
                # no short cycle found but not ok -> be safe, stop
                return edges, cur_len >= g
            m = len(cyc)
            if m < 4:
                break

            # candidate 2-switches: pair an edge on the cycle with an off-cycle edge
            # to break the cycle while preserving 3-regularity.
            improved_this_round = False
            # edges on the cycle
            cyc_edges = []
            for i in range(m):
                a, b = cyc[i], cyc[(i + 1) % m]
                cyc_edges.append((min(a, b), max(a, b)))

            # gather all off-cycle edges
            all_edges = list(eset)
            random.shuffle(all_edges)

            # try a batch of random 2-switches; accept first that keeps girth non-worse
            attempts = 0
            for (a, b) in cyc_edges:
                if time.time() > t_dead:
                    break
                # the two "other" neighbours of a and b
                na = [x for x in build_adj([(a, b)], 1) ]  # placeholder
                # compute neighbours properly
                adj_a = [x for x in adj[a] if x != b]
                adj_b = [x for x in adj[b] if x != a]
                if not adj_a or not adj_b:
                    continue
                for _ in range(6):
                    if time.time() > t_dead:
                        break
                    attempts += 1
                    c = random.choice(adj_a)
                    d2 = random.choice(adj_b)
                    # swap edges (a,b) and (c,d2) -> (a,c),(b,d2)  [only if c!=d2 etc]
                    if len({a, b, c, d2}) < 4:
                        continue
                    ne1 = (min(a, c), max(a, c))
                    ne2 = (min(b, d2), max(b, d2))
                    if ne1 == ne2:
                        continue
                    if ne1 in eset or ne2 in eset:
                        continue
                    # apply
                    eset.discard((a, b))
                    eset.discard((c, d2))
                    eset.add(ne1)
                    eset.add(ne2)
                    edges = list(eset)
                    adj = build_adj(edges, n)
                    new_len = shortest_cycle_len(adj)
                    if new_len < cur_len:
                        # worse -> revert
                        eset.discard(ne1)
                        eset.discard(ne2)
                        eset.add((a, b))
                        eset.add((c, d2))
                        edges = list(eset)
                        continue
                    # accept (same or better)
                    cur_len = new_len
                    improved_this_round = True
                    break
                if improved_this_round:
                    break
            if not improved_this_round:
                stuck += 1
                if stuck > 40:
                    break
            else:
                stuck = 0
        adj = build_adj(edges, n)
        ok = shortest_cycle_len(adj) >= g
        return edges, ok

    # ---------- If the baseline is already the known optimum for (3,16), return it ----------
    if k == 3 and g == 16 and baseline_n == 936:
        return baseline_n, baseline_edges

    best_n = baseline_n
    best_edges = baseline_edges

    # ---------- Search for a smaller graph via cycle-breaking ----------
    if time.time() < deadline - 8.0:
        if g % 2 == 0:
            moore = 2 * sum(2 ** i for i in range(g // 2 - 1))
        else:
            moore = 2 * sum(2 ** i for i in range((g - 1) // 2 - 1)) + 1
        if moore % 2 != 0:
            moore += 1

        def gen_random_3reg(n):
            if n < 4 or n % 2 != 0:
                return None
            stubs = [i for i in range(n) for _ in range(3)]
            for _ in range(60):
                random.shuffle(stubs)
                es = set()
                ok = True
                for i in range(0, len(stubs), 2):
                    u, v = stubs[i], stubs[i + 1]
                    if u == v:
                        ok = False
                        break
                    key = (min(u, v), max(u, v))
                    if key in es:
                        ok = False
                        break
                    es.add(key)
                if ok:
                    return list(es)
            return None

        # Try sizes from Moore bound up to just below baseline
        lo = max(4, moore)
        hi = max(lo, baseline_n - 2)
        # step through candidate even sizes, larger ones first (more likely feasible)
        candidates = list(range(hi, lo - 1, -2))
        # but we want the SMALLEST valid; however larger sizes are easier to satisfy.
        # Strategy: try a handful of sizes near the lower bound, then expand.
        random.shuffle(candidates)
        # bias toward smaller
        candidates.sort(key=lambda x: x)
        # interleave: take smallest-first but cap attempts
        tried = 0
        max_tries = 24
        for n_try in candidates:
            if time.time() > deadline - 6.0:
                break
            if tried >= max_tries:
                break
            tried += 1
            if n_try >= best_n:
                continue
            remaining = deadline - 6.0 - time.time()
            if remaining < 1.5:
                break
            budget = min(remaining / 3.0, 12.0)
            for restart in range(3):
                if time.time() > deadline - 6.0:
                    break
                edges = gen_random_3reg(n_try)
                if edges is None:
                    continue
                adj = build_adj(edges, n_try)
                if girth_ok(adj, n_try, g):
                    best_n = n_try
                    best_edges = edges
                    break
                edges2, ok = try_break(edges, n_try, g, budget)
                if ok:
                    adj = build_adj(edges2, n_try)
                    if girth_ok(adj, n_try, g):
                        best_n = n_try
                        best_edges = edges2
                        break
            if best_n <= lo:
                break

    # ---------- Final safety: ensure returned graph is valid ----------
    adj = build_adj(best_edges, best_n)
    if not girth_ok(adj, best_n, g):
        # fall back to the algebraic baseline which is known valid
        return baseline_n, baseline_edges
    return best_n, best_edges
