import time
import sys
import random
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # safety margin vs 120s limit

    # Moore bound
    def moore_bound(k, g):
        if g % 2 == 0:
            h = g // 2 - 1
            return k * ((k - 1) ** h - 1) // (k - 2) + 1
        else:
            h = (g - 1) // 2
            return 1 + k * ((k - 1) ** h - 1) // (k - 2)

    mb = moore_bound(k, g)

    # ---------- LUW construction (guaranteed fallback) ----------
    def luw_construct(k, g):
        q = k
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            def gf_add(x, y): return x ^ y
            def gf_sub(x, y): return x ^ y
            def gf_mul(x, y): return table[x][y]
        else:
            def gf_add(x, y): return (x + y) % q
            def gf_sub(x, y): return (x - y) % q
            def gf_mul(x, y): return x * y % q

        d = max(3, g - 5 if (g - 5) % 2 else g - 4)

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
                    if side == 0:
                        y[c] = gf_add(x[c], gf_mul(y[a], x[b]))
                    else:
                        y[c] = gf_sub(x[c], gf_mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []
        queue = [0]
        head = 0
        while head < len(order):
            if time.time() > deadline - 2:
                break
            v_idx = queue[head]
            head += 1
            v = order[v_idx]
            for other in neighbours(v):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                    queue.append(len(order) - 1)
                if v[0] == 0:
                    edges.append((v_idx, index[other]))
        return len(order), edges

    luw_n, luw_edges = luw_construct(k, g)

    # ---------- Girth verification (non-backtracking walks from every vertex) ----------
    def check_girth(n, adj, g):
        for s in range(n):
            dist = {}
            q2 = deque()
            for v in adj[s]:
                q2.append((v, s))
                dist[(v, s)] = 1
            while q2:
                cur, prev = q2.popleft()
                d_cur = dist[(cur, prev)]
                if d_cur >= g - 1:
                    continue
                for nxt in adj[cur]:
                    if nxt == prev:
                        continue
                    if nxt == s:
                        if d_cur + 1 < g:
                            return False
                    else:
                        if (nxt, cur) not in dist:
                            dist[(nxt, cur)] = d_cur + 1
                            q2.append((nxt, cur))
        return True

    # Verify LUW (it should be valid, but be safe)
    adj_luw = [[] for _ in range(luw_n)]
    for u, v in luw_edges:
        adj_luw[u].append(v)
        adj_luw[v].append(u)
    luw_ok = check_girth(luw_n, adj_luw, g)

    if not luw_ok:
        return luw_n, luw_edges

    best_n = luw_n
    best_edges = luw_edges

    # ---------- Fast circulant search using non-backtracking coefficient set ----------
    # A 4-regular circulant graph C(n, {a, b, n-a, n-b}) has girth >= g iff
    # no non-backtracking walk of length < g sums to 0 mod n.
    # Precompute the set of (c1, c2) coefficients for non-backtracking walks of length 3..g-1.
    valid_coeffs = set()
    if g > 3:
        states = set()
        states.add((1, 1, 0))
        states.add((-1, -1, 0))
        states.add((2, 0, 1))
        states.add((-2, 0, -1))
        for L in range(2, g):
            new_states = set()
            for last_step, c1, c2 in states:
                for step in (1, -1, 2, -2):
                    if step == -last_step:
                        continue
                    nc1 = c1 + (1 if step == 1 else -1 if step == -1 else 0)
                    nc2 = c2 + (1 if step == 2 else -1 if step == -2 else 0)
                    new_states.add((step, nc1, nc2))
            states = new_states
            if L >= 3:
                for _, c1, c2 in states:
                    valid_coeffs.add((c1, c2))

    def circulant_girth_ok(n, a, b):
        for c1, c2 in valid_coeffs:
            if (c1 * a + c2 * b) % n == 0:
                return False
        return True

    def build_circulant_edges(n, a, b):
        edges = set()
        for i in range(n):
            j1 = (i + a) % n
            edges.add((i, j1) if i < j1 else (j1, i))
            j2 = (i + b) % n
            edges.add((i, j2) if i < j2 else (j2, i))
        return list(edges)

    # Search n from Moore bound upward for a valid circulant
    if best_n > mb:
        n_max_search = min(best_n - 1, mb + 300)
        found_circ = False
        for n in range(mb, n_max_search + 1):
            if time.time() > deadline - 5:
                break
            # Deterministic sweep of small offsets first
            half = n // 2 - 1
            max_off = min(200, half)
            for a in range(1, max_off + 1):
                if time.time() > deadline - 5:
                    break
                for b in range(a + 1, max_off + 1):
                    if circulant_girth_ok(n, a, b):
                        best_n = n
                        best_edges = build_circulant_edges(n, a, b)
                        found_circ = True
                        break
                if found_circ:
                    break
            if found_circ:
                break
            # Random offsets for this n
            for _ in range(50):
                if time.time() > deadline - 5:
                    break
                a = random.randint(1, half)
                b = random.randint(1, half)
                if a == b:
                    continue
                if circulant_girth_ok(n, a, b):
                    best_n = n
                    best_edges = build_circulant_edges(n, a, b)
                    found_circ = True
                    break
            if found_circ:
                break

    # ---------- Secondary: random configuration-model search for non-circulant graphs ----------
    if best_n > mb and time.time() < deadline - 10:
        n_max_rand = min(best_n - 1, mb + 100)
        for n in range(mb, n_max_rand + 1):
            if time.time() > deadline - 10:
                break
            found_rand = False
            for trial in range(30):
                if time.time() > deadline - 10:
                    break
                random.seed(trial * n + 12345)
                stubs = []
                for i in range(n):
                    for _ in range(k):
                        stubs.append(i)
                random.shuffle(stubs)
                edges_set = set()
                valid = True
                for i in range(0, len(stubs), 2):
                    u, v = stubs[i], stubs[i + 1]
                    if u == v:
                        valid = False
                        break
                    key = (u, v) if u < v else (v, u)
                    if key in edges_set:
                        valid = False
                        break
                    edges_set.add(key)
                if not valid:
                    continue
                adj2 = [[] for _ in range(n)]
                for u, v in edges_set:
                    adj2[u].append(v)
                    adj2[v].append(u)
                if any(len(adj2[v]) != k for v in range(n)):
                    continue
                if check_girth(n, adj2, g):
                    best_n = n
                    best_edges = list(edges_set)
                    found_rand = True
                    break
            if found_rand:
                break

    return best_n, best_edges
# EVOLVE-BLOCK-END
