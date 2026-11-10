import time
import random
from collections import deque

import numpy as np

# ---------------------------------------------------------------------------
# Fallback: Lazebnik-Ustimenko-Woldar construction (known valid, large)
# ---------------------------------------------------------------------------
def _luw_construct(k, g):
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


# ---------------------------------------------------------------------------
# Generic girth certificate via non-backtracking BFS (used for small graphs)
# ---------------------------------------------------------------------------
def _check_girth(n, edges, g):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    half = g // 2
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            if dist[u] >= half:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and dist[v] <= dist[u]:
                    if dist[u] + dist[v] + 1 < g:
                        return False
    return True


# ---------------------------------------------------------------------------
# Fast K4 voltage construction (small n = 4m); used as a quick first attempt
# ---------------------------------------------------------------------------
def _try_voltage(deadline):
    random.seed(1234)
    for m in range(128, 200):
        if time.time() > deadline - 15:
            return None
        n = 4 * m
        base = [
            (0 * m + t, 1 * m + (t + a) % m) for t in range(m)
        ]
        for _ in range(150):
            if time.time() > deadline - 10:
                break
            a, b, c, d, e, f = (random.randint(1, m - 1) for _ in range(6))
            edges = []
            for t in range(m):
                edges.append((0 * m + t, 1 * m + (t + a) % m))
                edges.append((0 * m + t, 2 * m + (t + b) % m))
                edges.append((0 * m + t, 3 * m + (t + c) % m))
                edges.append((1 * m + t, 2 * m + (t + d) % m))
                edges.append((1 * m + t, 3 * m + (t + e) % m))
                edges.append((2 * m + t, 3 * m + (t + f) % m))
            if _check_girth(n, edges, 16):
                return n, edges
    return None


# ---------------------------------------------------------------------------
# Cayley graph on SL(2, q) with a fast word-based girth certificate
# ---------------------------------------------------------------------------
def _sl2_cayley(q, deadline):
    # ---- GF(q) setup (q = 2^e, e in 4..6) ----
    e = int(np.log2(q))
    # find a primitive polynomial of degree e over GF(2)
    def is_primitive(poly, e):
        # poly: int, bit e is 1 (monic). Represent as integer with bit i = coeff of x^i
        def gf_mul(a, b):
            r = 0
            for _ in range(e):
                if b & 1:
                    r ^= a
                b >>= 1
                a <<= 1
                if a & (1 << e):
                    a ^= poly
            return r
        # generator: x (i.e. value 2). Check x^(q-1)=1 and order exactly q-1
        x = 2
        cur = x
        for _ in range(q - 2):
            cur = gf_mul(cur, x)
        if gf_mul(cur, x) != 1:
            return False
        # check no smaller divisor
        n = q - 1
        # factor n
        m = n
        factors = []
        p = 2
        while p * p <= m:
            if m % p == 0:
                factors.append(p)
                while m % p == 0:
                    m //= p
            p += 1
        if m > 1:
            factors.append(m)
        for pf in factors:
            cur = 1
            for _ in range(n // pf):
                cur = gf_mul(cur, x)
            if cur == 1:
                return False
        return True

    poly = None
    for cand in range(1 << (e - 1), 1 << e):  # monic, degree e
        # ensure not zero poly; test primitive
        if is_primitive(cand, e):
            poly = cand
            break
    if poly is None:
        poly = 0b10011  # x^4 + x + 1 fallback

    # precompute multiplication table for GF(q)
    def gf_mul(a, b):
        r = 0
        for _ in range(e):
            if b & 1:
                r ^= a
            b >>= 1
            a <<= 1
            if a & (1 << e):
                a ^= poly
        return r

    # multiplication table as int array
    mul = np.zeros((q, q), dtype=np.int32)
    for a in range(q):
        for b in range(q):
            mul[a, b] = gf_mul(a, b)
    # inverse table
    inv = np.zeros(q, dtype=np.int32)
    for a in range(1, q):
        for b in range(1, q):
            if gf_mul(a, b) == 1:
                inv[a] = b
                break

    # ---- Enumerate SL(2, q): matrices [a b; c d] with ad + bc = 1 (char 2) ----
    group = []
    gmap = {}
    for a in range(q):
        for b in range(q):
            for c in range(q):
                if a == 0:
                    if gf_mul(b, c) == 1:
                        for d in range(q):
                            m = (a, b, c, d)
                            if m not in gmap:
                                gmap[m] = len(group)
                                group.append(m)
                else:
                    rhs = gf_mul(b, c) ^ 1
                    d = gf_mul(rhs, inv[a])
                    m = (a, b, c, d)
                    if m not in gmap:
                        gmap[m] = len(group)
                        group.append(m)
    n = len(group)
    if n > 60000:
        return None

    # matrix multiplication
    def mat_mul(m1, m2):
        a1, b1, c1, d1 = m1
        a2, b2, c2, d2 = m2
        a = gf_mul(a1, a2) ^ gf_mul(b1, c2)
        b = gf_mul(a1, b2) ^ gf_mul(b1, d2)
        c = gf_mul(c1, a2) ^ gf_mul(d1, c2)
        d = gf_mul(c1, b2) ^ gf_mul(d1, d2)
        return (a, b, c, d)

    # precompute multiplication table (index-based)
    print(f"  SL(2,{q}): n={n}, building mul table...")
    t0 = time.time()
    mul_table = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        m1 = group[i]
        for j in range(n):
            res = mat_mul(m1, group[j])
            mul_table[i, j] = gmap[res]
    print(f"  mul table done in {time.time()-t0:.1f}s")

    # identity index
    ident = gmap[(1, 0, 0, 1)]

    # ---- Girth certificate: no non-backtracking word of length <= 7 equals identity ----
    def has_short_word(s1, s2, s3):
        # words are sequences of generator indices (0,1,2); non-backtracking means
        # w[t] != 3 - w[t-1] (inverse of previous). Since generators may not be
        # self-inverse, we must track the actual group element and the previous
        # generator used, and forbid immediately applying the inverse of the last.
        # Simpler and exact: a non-backtracking walk in the Cayley graph forbids
        # returning along the same edge, i.e. next generator != inverse of last
        # generator. We track (current_index, last_gen).
        # Start: (ident, -1). Expand up to depth 7.
        # State: (idx, last). last in {-1,0,1,2}.
        seen = set()
        frontier = [(ident, -1)]
        for depth in range(7):
            new = []
            for idx, last in frontier:
                for g in range(3):
                    if g == 3 - last and last != -1:
                        # backtracking: applying inverse of last generator
                        # but only if generators are self-inverse OR we track properly.
                        # To be safe, treat "backtrack" as applying the inverse of
                        # the last generator. We need inverse generator index.
                        pass
                # handle below
            frontier = new
        # We'll implement properly with inverse tracking.
        return False

    # We need inverse of each generator to define backtracking.
    def mat_inv(m):
        a, b, c, d = m
        # det=1, inverse = [d -b; -c a] = [d b; c a] in char 2
        return (d, b, c, a)

    def check_girth_word(s1, s2, s3):
        gens = [s1, s2, s3]
        inv_gens = [gmap[mat_inv(group[s])] for s in gens]
        # state: (idx, last_gen_index or -1)
        # backtracking: next gen g is forbidden if g == inv_gens[last] when last != -1
        seen = set()
        frontier = [(ident, -1)]
        for depth in range(7):
            new = []
            for idx, last in frontier:
                for g in range(3):
                    if last != -1 and g == inv_gens[last]:
                        continue  # non-backtracking
                    nidx = int(mul_table[idx, gens[g]])
                    if nidx == ident and depth + 1 >= 2:
                        # a non-trivial non-backtracking word equals identity
                        return False
                    st = (nidx, g)
                    if st not in seen:
                        seen.add(st)
                        new.append(st)
            frontier = new
        return True

    # ---- Search for a good generating set ----
    random.seed(7)
    all_idx = list(range(n))
    # self-inverse candidates (M^2 = I, M != I)
    self_inv = []
    for i in range(n):
        if i == ident:
            continue
        m = group[i]
        if mat_mul(m, m) == (1, 0, 0, 1):
            self_inv.append(i)

    print(f"  self-inverse count: {len(self_inv)}")

    for trial in range(3000):
        if time.time() > deadline - 8:
            break
        # Mix strategies
        if trial % 3 == 0 and len(self_inv) >= 3:
            s1, s2, s3 = random.sample(self_inv, 3)
        elif trial % 3 == 1 and len(self_inv) >= 1:
            s1 = random.choice(self_inv)
            s2 = random.choice(all_idx)
            s3 = gmap[mat_inv(group[s2])]
            if s2 == s3:
                continue
        else:
            s1 = random.choice(all_idx)
            s2 = random.choice(all_idx)
            s3 = random.choice(all_idx)
            # ensure not identity and distinct-ish
            if s1 == ident or s2 == ident or s3 == ident:
                continue
            if s1 == s2 or s2 == s3 or s1 == s3:
                continue
        if check_girth_word(s1, s2, s3):
            print(f"  found generating set at trial {trial}: {s1},{s2},{s3}")
            edges = []
            for i in range(n):
                edges.append((i, int(mul_table[i, s1])))
                edges.append((i, int(mul_table[i, s2])))
                edges.append((i, int(mul_table[i, s3])))
            return n, edges
    return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115

    if k != 3 or g != 16:
        return _luw_construct(k, g)

    # 1) Quick K4 voltage attempt (small n)
    res = _try_voltage(deadline)
    if res is not None:
        return res

    # 2) Cayley graph on SL(2, q) for a few field sizes
    for q in (16, 32, 64):
        if time.time() > deadline - 20:
            break
        try:
            res = _sl2_cayley(q, deadline)
            if res is not None:
                return res
        except Exception:
            pass

    # 3) Fallback
    return _luw_construct(k, g)
