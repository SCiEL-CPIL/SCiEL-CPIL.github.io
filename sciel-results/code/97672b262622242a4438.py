import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    t0 = time.time()
    DEADLINE = t0 + 145.0

    def rem():
        return DEADLINE - time.time()

    # ---------- fast girth check via BFS from every vertex ----------
    def check_girth(n, edges, min_girth, limit):
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        best = 1 << 30
        for s in range(n):
            if time.time() > limit:
                return -1
            dist = [-1] * n
            par = [-1] * n
            dist[s] = 0
            q = [s]
            qi = 0
            while qi < len(q):
                node = q[qi]
                qi += 1
                nd = dist[node] + 1
                for nb in adj[node]:
                    if nb == par[node]:
                        continue
                    if dist[nb] == -1:
                        dist[nb] = nd
                        par[nb] = node
                        q.append(nb)
                    else:
                        c = dist[node] + dist[nb] + 1
                        if c < best:
                            best = c
                        if c < min_girth:
                            return c
        return best

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

    # ---------- Voltage lift of the Petersen graph ----------
    def voltage_lift_petersen():
        if k != 3 or g != 16:
            return None

        pet = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
               (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
               (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)]
        n_base = 10
        eidx = {}
        for i, (u, v) in enumerate(pet):
            eidx[(u, v)] = i
            eidx[(v, u)] = i
        adj = [[] for _ in range(n_base)]
        for u, v in pet:
            adj[u].append(v)
            adj[v].append(u)

        # All non-backtracking closed walks of length 5..15 -> net edge count vectors
        maxL = g - 1
        vecs = set()
        for s in range(n_base):
            stack = [(s, -1, 0, [0] * 15)]
            while stack:
                cur, prev, L, counts = stack.pop()
                if L >= maxL:
                    continue
                for nxt in adj[cur]:
                    if nxt == prev:
                        continue
                    ei = eidx[(cur, nxt)]
                    u, v = pet[ei]
                    d = 1 if cur == u else -1
                    nc = counts.copy()
                    nc[ei] += d
                    nl = L + 1
                    if nxt == s and nl >= 5:
                        vecs.add(tuple(nc))
                    stack.append((nxt, cur, nl, nc))

        uniq = set()
        for c in vecs:
            if all(x == 0 for x in c):
                continue
            f = -1
            for i in range(15):
                if c[i] != 0:
                    f = i
                    break
            if f != -1 and c[f] < 0:
                c = tuple(-x for x in c)
            uniq.add(c)
        C = np.array(list(uniq), dtype=np.int64)
        if C.size == 0:
            return None
        Cn = C.shape[0]

        def solve(m, max_iters=50000):
            Cmod = C % m
            v = np.random.randint(1, m, size=15).astype(np.int64)
            res = (Cmod @ v) % m
            best = int(np.sum(res == 0))
            if best == 0:
                return v
            T = 1.0
            cool = 0.99995
            for it in range(max_iters):
                if rem() < 8:
                    break
                r = random.random()
                if r < 0.5:
                    j = random.randint(0, 14)
                    ov = v[j]
                    v[j] = random.randint(1, m - 1)
                    res = (Cmod @ v) % m
                    nc = int(np.sum(res == 0))
                    if nc < best or (T > 0.02 and random.random() < np.exp(-(nc - best) / T)):
                        if nc < best:
                            best = nc
                        if nc == 0:
                            return v
                    else:
                        v[j] = ov
                else:
                    j1 = random.randint(0, 14)
                    j2 = random.randint(0, 14)
                    if j1 == j2:
                        continue
                    o1, o2 = v[j1], v[j2]
                    v[j1] = o2
                    v[j2] = o1
                    res = (Cmod @ v) % m
                    nc = int(np.sum(res == 0))
                    if nc < best or (T > 0.02 and random.random() < np.exp(-(nc - best) / T)):
                        if nc < best:
                            best = nc
                        if nc == 0:
                            return v
                    else:
                        v[j1] = o1
                        v[j2] = o2
                T *= cool
            return None

        def build(v, m):
            n = 10 * m
            edges = []
            for u, vb in pet:
                vol = v[eidx[(u, vb)]]
                for layer in range(m):
                    a = u * m + layer
                    b = vb * m + (layer + vol) % m
                    edges.append((a, b) if a < b else (b, a))
            return n, edges

        primes = [53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
                  101, 103, 107, 109, 113, 127, 131, 137, 139, 149,
                  151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199,
                  211, 223, 227, 229, 233, 239, 241, 251, 257, 263,
                  269, 271, 277, 281, 283, 293, 307, 311, 313, 317,
                  331, 337, 347, 349, 353, 359, 367, 373, 379, 383,
                  389, 397, 401, 409, 419, 421, 431, 433, 439, 443,
                  449, 457, 461, 463, 467, 479, 487, 491, 499, 503]

        for m in primes:
            if rem() < 20:
                break
            n_candidate = 10 * m
            if n_candidate < 512:
                continue
            for attempt in range(50):
                if rem() < 12:
                    break
                v = solve(m)
                if v is not None:
                    n, edges = build(v, m)
                    gc = check_girth(n, edges, g, DEADLINE - 5)
                    if gc >= g:
                        return (n, edges)
        return None

    # ---------- LUW (Lubell-Ulrich-Wilson) construction ----------
    def luw_fallback(k, g):
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

    # ---------- 2-switch girth improvement heuristic ----------
    def two_switch_improve(n, edges, target_girth, limit):
        """
        Start from a 3-regular graph and try to increase girth via 2-switches.
        A 2-switch picks two edges (a,b) and (c,d) and replaces them with (a,c) and (b,d)
        (or (a,d) and (b,c)), maintaining 3-regularity.
        """
        if time.time() > limit:
            return n, edges

        # Build adjacency
        adj = [set() for _ in range(n)]
        for i, j in edges:
            adj[i].add(j)
            adj[j].add(i)

        def get_edges():
            result = []
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        result.append((i, j))
            return result

        def find_short_cycle(start, max_len):
            """Find a cycle of length < max_len from start using BFS."""
            dist = [-1] * n
            par = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                node = q.popleft()
                if dist[node] >= max_len:
                    continue
                for nb in adj[node]:
                    if nb == par[node]:
                        continue
                    if dist[nb] == -1:
                        dist[nb] = dist[node] + 1
                        par[nb] = node
                        q.append(nb)
                    else:
                        cycle_len = dist[node] + dist[nb] + 1
                        if cycle_len < max_len and cycle_len >= 3:
                            # Reconstruct cycle
                            path1 = []
                            cur = node
                            while cur != -1:
                                path1.append(cur)
                                cur = par[cur]
                            path2 = []
                            cur = nb
                            while cur != -1:
                                path2.append(cur)
                                cur = par[cur]
                            # Find LCA
                            set1 = set(path1)
                            lca = None
                            for v in path2:
                                if v in set1:
                                    lca = v
                                    break
                            if lca is not None:
                                idx1 = path1.index(lca)
                                idx2 = path2.index(lca)
                                cyc = path1[idx1:] + path2[idx2:][::-1]
                                # Remove the LCA duplicate at end
                                if len(cyc) > 1 and cyc[0] == cyc[-1]:
                                    cyc = cyc[:-1]
                                if len(cyc) < max_len and len(cyc) >= 3:
                                    return cyc
            return None

        def do_two_switch(e1, e2, variant):
            """Perform a 2-switch. e1=(a,b), e2=(c,d). variant 0: (a,c)+(b,d), variant 1: (a,d)+(b,c)"""
            a, b = e1
            c, d = e2
            # Check validity: no self-loops, no multi-edges
            if variant == 0:
                new_edges = [(a, c), (b, d)]
            else:
                new_edges = [(a, d), (b, c)]
            for x, y in new_edges:
                if x == y:
                    return False
                if y in adj[x]:
                    return False
            # Apply
            adj[a].discard(b)
            adj[b].discard(a)
            adj[c].discard(d)
            adj[d].discard(c)
            for x, y in new_edges:
                adj[x].add(y)
                adj[y].add(x)
            return True

        def undo_two_switch(e1, e2, variant):
            a, b = e1
            c, d = e2
            if variant == 0:
                adj[a].discard(c)
                adj[c].discard(a)
                adj[b].discard(d)
                adj[d].discard(b)
                adj[a].add(b)
                adj[b].add(a)
                adj[c].add(d)
                adj[d].add(c)
            else:
                adj[a].discard(d)
                adj[d].discard(a)
                adj[b].discard(c)
                adj[c].discard(b)
                adj[a].add(b)
                adj[b].add(a)
                adj[c].add(d)
                adj[d].add(c)

        # Iteratively find and eliminate short cycles
        for iteration in range(2000):
            if time.time() > limit:
                break
            # Find a short cycle
            found = False
            for start in range(0, n, max(1, n // 200)):
                if time.time() > limit:
                    break
                cyc = find_short_cycle(start, target_girth)
                if cyc is not None and len(cyc) >= 3:
                    # Find two edges on the cycle to switch
                    # Pick two non-adjacent edges on the cycle
                    m = len(cyc)
                    e1 = (cyc[0], cyc[1])
                    e2 = (cyc[m // 2], cyc[(m // 2 + 1) % m])
                    # Try both variants
                    for variant in [0, 1]:
                        # Check the other endpoints
                        if variant == 0:
                            # (a,c) and (b,d) where e1=(a,b), e2=(c,d)
                            a, b = e1
                            c, d = e2
                            if c == a or c == b or d == a or d == b:
                                continue
                        else:
                            a, b = e1
                            c, d = e2
                            if d == a or d == b or c == a or c == b:
                                continue
                        if do_two_switch(e1, e2, variant):
                            found = True
                            break
                    if found:
                        break
            if not found:
                break

        return n, get_edges()

    # ---------- Main logic ----------
    if k == 3 and g == 16:
        r = voltage_lift_petersen()
        if r is not None:
            return r
        # Fallback: LUW + 2-switch improvement
        n, edges = luw_fallback(k, g)
        if rem() > 30:
            n, edges = two_switch_improve(n, edges, g, DEADLINE - 5)
        return n, edges

    return luw_fallback(k, g)
