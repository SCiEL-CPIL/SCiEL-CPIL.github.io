"""Construct a k-regular graph of girth at least g, minimizing vertex count."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill

    def time_left():
        return deadline - time.time()

    # Moore bound for lower bound on n
    if g % 2 == 0:
        moore = k + k * (k - 1) ** (g // 2 - 1)
    else:
        if k == 2:
            moore = g
        else:
            h = (g - 1) // 2
            if k > 2:
                moore = 1 + k * ((k - 1) ** h - 1) // (k - 2)
            else:
                moore = g

    best_n = None
    best_edges = None

    def build_adj(edges, n):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            if u == v:
                return None
            adj[u].add(v)
            adj[v].add(u)
        for i in range(n):
            if len(adj[i]) != len(adj[i]):
                return None
        return [sorted(a) for a in adj]

    def nonback_girth(adj, n):
        """Return length of shortest cycle via non-backtracking walks from every vertex."""
        # state (v, p): arrived at v from p
        prev = {}
        q = deque()
        for u in range(n):
            for p in adj[u]:
                key = (p, u)
                prev[key] = (u, 1)
                q.append(key)
        while q:
            (v, p), (u, dist) = q.popleft()
            for w in adj[v]:
                if w == p:
                    continue
                key = (w, v)
                if w == u:
                    return dist + 1
                if key in prev:
                    continue
                prev[key] = (v, dist + 1)
                q.append(key)
        return 10**9

    def verify(n, edges, min_girth):
        """Return (ok, girth). ok iff simple, k-regular, girth >= min_girth."""
        if n == 0:
            return False, 0
        adj = [set() for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            if u == v:
                return False, 0
            a, b = (u, v) if u < v else (v, u)
            if a in edge_set:
                return False, 0
            edge_set.add((a, b))
            adj[u].add(v)
            adj[v].add(u)
        for i in range(n):
            if len(adj[i]) != k:
                return False, 0
        adj = [sorted(a) for a in adj]
        girth = nonback_girth(adj, n)
        return girth >= min_girth, girth

    def luw_construct():
        """Lazebnik-Ustimenko-Woldar construction."""
        q = k
        if (g - 5) % 2 == 1:
            d = max(3, g - 5)
        else:
            d = max(3, g - 4)
        if q == 4:
            table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    def cayley_sl2_11():
        """Deterministic Cayley graph over SL(2,11) with 5 involutions (or small-order
        elements) as connection set. Yields a 5-regular graph on 1320 vertices."""
        # SL(2,11) has order 1320. Represent elements as 2x2 matrices mod 11 with det=1.
        # We'll enumerate all elements, then pick 5 involutions that generate the group
        # and produce a graph with no short relations (girth >= 9).
        #
        # Enumeration: all (a,b,c,d) in Z_11^4 with ad-bc = 1 mod 11.
        # That's 1320 elements.
        p = 11
        elements = []
        for a in range(p):
            for b in range(p):
                for c in range(p):
                    # ad - bc = 1 => ad = 1 + bc mod p
                    # if a != 0, d = (1+bc)*a^{-1}
                    # if a == 0, then -bc = 1 => bc = -1 = 10, so b != 0 and c = 10*b^{-1}
                    if a != 0:
                        inv_a = pow(a, p - 2, p)
                        d = (1 + b * c) * inv_a % p
                        elements.append((a, b, c, d))
                    else:
                        # -bc = 1 => bc = 10
                        for bb in range(1, p):
                            inv_bb = pow(bb, p - 2, p)
                            cc = 10 * inv_bb % p
                            if b == bb:
                                elements.append((0, bb, cc, 0))
        # Deduplicate (the a==0 branch may add duplicates)
        elements = list(set(elements))
        n = len(elements)  # should be 1320
        if n != 1320:
            return None

        # Map element -> index
        idx = {e: i for i, e in enumerate(elements)}

        def mat_mul(m1, m2):
            a1, b1, c1, d1 = m1
            a2, b2, c2, d2 = m2
            return ((a1*a2 + b1*c2) % p,
                    (a1*b2 + b1*d2) % p,
                    (c1*a2 + d1*c2) % p,
                    (c1*b2 + d1*d2) % p)

        def mat_inv(m):
            a, b, c, d = m
            # det = 1, so inv = (d, -b, -c, a)
            return (d, (-b) % p, (-c) % p, a)

        # Find all involutions (order 2, not identity)
        identity = (1, 0, 0, 1)
        involutions = []
        for e in elements:
            if e == identity:
                continue
            e2 = mat_mul(e, e)
            if e2 == identity:
                involutions.append(e)

        # We need 5 involutions that generate SL(2,11) and whose Cayley graph has girth >= 9.
        # Strategy: pick involutions that are "far apart" to avoid short relations.
        # A known good set for SL(2,11) Cayley graph with girth >= 9:
        # We'll try several combinations and verify.
        #
        # First, pick a generating set of 5 involutions.
        # Let's try: s1, s2, s3, s4, s5 where we select involutions that
        # are spread out in the group.

        # Heuristic: pick involutions with distinct "types"
        # Group involutions by trace (a+d mod 11)
        by_trace = {}
        for e in involutions:
            tr = (e[0] + e[3]) % p
            by_trace.setdefault(tr, []).append(e)

        # Pick one from each trace class to maximize spread
        chosen = []
        used_traces = set()
        for tr in sorted(by_trace.keys()):
            if tr in used_traces:
                continue
            for e in by_trace[tr]:
                if e not in chosen:
                    chosen.append(e)
                    used_traces.add(tr)
                    break
            if len(chosen) == 5:
                break

        # If we don't have 5 distinct traces, fill in
        if len(chosen) < 5:
            for e in involutions:
                if e not in chosen:
                    chosen.append(e)
                if len(chosen) == 5:
                    break

        if len(chosen) < 5:
            return None

        conn_set = chosen[:5]

        # Build Cayley graph
        adj = [[] for _ in range(n)]
        edges = []
        for i, v in enumerate(elements):
            for s in conn_set:
                w = mat_mul(v, s)
                j = idx[w]
                if i < j:
                    edges.append((i, j))
                adj[i].append(j)

        # Verify: 5-regular, simple, girth >= g
        ok, girth = verify(n, edges, g)
        if ok:
            return n, edges

        # Try other combinations of involutions if the first didn't work
        # Sample a few random 5-subsets
        random.seed(42)
        for attempt in range(200):
            if time_left() < 10:
                break
            combo = random.sample(involutions, 5)
            adj2 = [[] for _ in range(n)]
            edges2 = []
            for i, v in enumerate(elements):
                for s in combo:
                    w = mat_mul(v, s)
                    j = idx[w]
                    if i < j:
                        edges2.append((i, j))
                    adj2[i].append(j)
            ok2, girth2 = verify(n, edges2, g)
            if ok2:
                return n, edges2
        return None

    def random_regular(n, k_val, seed=None):
        """Generate a random k-regular graph on n vertices using configuration model."""
        if seed is not None:
            random.seed(seed)
        if n * k_val % 2 != 0:
            return None
        stubs = [i for i in range(n) for _ in range(k_val)]
        random.shuffle(stubs)
        adj = [set() for _ in range(n)]
        valid = True
        for i in range(n * k_val // 2):
            u, v = stubs[2 * i], stubs[2 * i + 1]
            if u == v or v in adj[u]:
                valid = False
                break
            adj[u].add(v)
            adj[v].add(u)
        if not valid:
            return None
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
        return n, edges

    def polish(n, edges, target_girth, max_iters=2000):
        """Try 2-switches to see if we can maintain girth >= target_girth.
        This doesn't reduce n but can validate robustness. We use it as a
        verification polish: if the graph is fragile, we might find issues."""
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        # Just verify once more with the non-backtracking method
        adj_sorted = [sorted(a) for a in adj]
        girth = nonback_girth(adj_sorted, n)
        return girth >= target_girth

    # === Main search ===

    # Step 1: Try explicit LUW construction (fast, deterministic)
    if time_left() > 15:
        try:
            n_exp, edges_exp = luw_construct()
            ok, girth_exp = verify(n_exp, edges_exp, g)
            if ok:
                best_n = n_exp
                best_edges = edges_exp
        except Exception:
            pass

    # Step 2: Try Cayley graph over SL(2,11) for k=5
    if k == 5 and time_left() > 30:
        try:
            result = cayley_sl2_11()
            if result is not None:
                n_cay, edges_cay = result
                if best_n is None or n_cay < best_n:
                    best_n = n_cay
                    best_edges = edges_cay
        except Exception:
            pass

    # Step 3: Try random regular graphs if we have time
    if time_left() > 20:
        # For k=5, g=9: try random graphs near known sizes
        # The LUW construction gives some n; try to beat it with random search
        if best_n is not None:
            upper = best_n
        else:
            upper = max(moore * 3, 500)

        # Try a few sizes below the current best
        search_sizes = []
        if best_n is not None:
            # Try sizes from moore to best_n - 1, but only a few
            step = max(1, (best_n - moore) // 10)
            for n_try in range(max(moore, k + 1), best_n, step):
                if n_try * k % 2 == 0:
                    search_sizes.append(n_try)
        else:
            for n_try in range(max(moore, k + 1), upper + 1, max(1, upper // 10)):
                if n_try * k % 2 == 0:
                    search_sizes.append(n_try)

        for n_try in search_sizes:
            if time_left() < 10:
                break
            for attempt in range(30):
                if time_left() < 10:
                    break
                result = random_regular(n_try, k, seed=random.randint(0, 2**31))
                if result is not None:
                    n_r, edges_r = result
                    ok, _ = verify(n_r, edges_r, g)
                    if ok:
                        if best_n is None or n_r < best_n:
                            best_n = n_r
                            best_edges = edges_r
                        break

    # Step 4: Polish - final verification
    if best_n is not None and best_edges is not None:
        # Final check
        ok_final, _ = verify(best_n, best_edges, g)
        if not ok_final:
            # Fallback to LUW
            try:
                n_exp, edges_exp = luw_construct()
                best_n, best_edges = n_exp, edges_exp
            except Exception:
                pass

    if best_n is None:
        # Absolute fallback
        n_exp, edges_exp = luw_construct()
        return n_exp, edges_exp

    return best_n, best_edges
# EVOLVE-BLOCK-END
