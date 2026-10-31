"""Structural redesign: parametric cubic bipartite girth-g graphs via affine-linear
neighbor maps over GF(q), searched randomly and verified by exact non-backtracking BFS."""
import time

_START = time.time()
_DEADLINE = _START + 140.0  # explicit wall-clock guard; keep 10s safety margin vs 150s kill


def _modinv(a, m):
    # extended Euclid, m prime
    t, nt = 0, 1
    r, nr = m, a % m
    while nr:
        q = r // nr
        t, nt = nt, t - q * nt
        r, nr = nr, r - q * nr
    return t % m


def construct(k, g):
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = _START + 140.0
    if k != 3:
        raise ValueError("only k=3 supported")
    q = 3  # smallest prime field -> smallest graph

    inv2 = _modinv(2, q)
    inv3 = _modinv(3, q)

    def add(v, w):
        return ((v[0] + w[0]) % q, (v[1] + w[1]) % q, (v[2] + w[2]) % q)

    def sub(v, w):
        return ((v[0] - w[0]) % q, (v[1] - w[1]) % q, (v[2] - w[2]) % q)

    def neg(v):
        return ((-v[0]) % q, (-v[1]) % q, (-v[2]) % q)

    def scale(v, s):
        return ((v[0] * s) % q, (v[1] * s) % q, (v[2] * s) % q)

    def matvec(M, v):
        return (
            (M[0][0] * v[0] + M[0][1] * v[1] + M[0][2] * v[2]) % q,
            (M[1][0] * v[0] + M[1][1] * v[1] + M[1][2] * v[2]) % q,
            (M[2][0] * v[0] + M[2][1] * v[1] + M[2][2] * v[2]) % q,
        )

    def rand_vec():
        return (
            (0 if q == 2 else (1 if (q == 3) else __import__("random").randint(0, q - 1)))
            if False else (0, 0, 0)
        )

    import random
    random.seed(12345)

    def rand_vec():
        if q == 2:
            return (random.randint(0, 1), random.randint(0, 1), random.randint(0, 1))
        if q == 3:
            return (random.randint(0, 2), random.randint(0, 2), random.randint(0, 2))
        return (random.randint(0, q - 1), random.randint(0, q - 1), random.randint(0, q - 1))

    def rand_mat():
        return [[random.randint(0, q - 1) for _ in range(3)] for _ in range(3)]

    def apply_map(v, M, t):
        return add(matvec(M, v), t)

    def check_girth(edges, n):
        # exact: no cycle shorter than g. Non-backtracking BFS from every vertex.
        adj = [[] for _ in range(n)]
        for a, b in edges:
            adj[a].append(b)
            adj[b].append(a)
        for s in range(n):
            # state (node, prev_node); depth. Start from each neighbour of s.
            seen = set()
            stack = []
            for nb in adj[s]:
                seen.add((nb, s))
                stack.append((nb, s, 1))
            while stack:
                u, p, depth = stack.pop()
                if depth >= g:
                    continue
                for v in adj[u]:
                    if v == p:
                        continue
                    if v == s:
                        return False  # closed non-backtracking walk of length depth+1 < g
                    if (v, u) in seen:
                        continue
                    seen.add((v, u))
                    stack.append((v, u, depth + 1))
        return True

    def build(M1, M2, M3, t1, t2, t3):
        # Build bipartite graph: side A = {0..26}, side B = {27..53} (n=54).
        # For each a in A, its three neighbors in B.
        n = 2 * q * q * q
        half = q * q * q
        # map a (0..half-1) -> vec
        def idx_to_vec(i):
            return (i % q, (i // q) % q, i // (q * q))

        def vec_to_idx(v):
            return v[0] + q * v[1] + q * q * v[2]

        edges = []
        degB = [0] * half
        for a in range(half):
            va = idx_to_vec(a)
            nb = [
                apply_map(va, M1, t1),
                apply_map(va, M2, t2),
                apply_map(va, M3, t3),
            ]
            # self-loop check: a vertex in A cannot map to itself in A (different side), fine.
            # repeated neighbor check
            if nb[0] == nb[1] or nb[0] == nb[2] or nb[1] == nb[2]:
                return None
            for j, w in enumerate(nb):
                b = vec_to_idx(w)
                edges.append((a, half + b))
                degB[b] += 1
        # B regularity
        for b in range(half):
            if degB[b] != 3:
                return None
        # simple: check no duplicate edges (A-degree already 3 distinct; B-degree 3)
        # duplicate edge would require same (a,b) twice -> a has distinct b, so fine.
        return n, edges

    def evaluate(M1, M2, M3, t1, t2, t3):
        res = build(M1, M2, M3, t1, t2, t3)
        if res is None:
            return None
        n, edges = res
        if not check_girth(edges, n):
            return None
        return n, edges

    best = None
    t0 = time.time()
    it = 0
    while time.time() < _DEADLINE:
        M1, M2, M3 = rand_mat(), rand_mat(), rand_mat()
        t1, t2, t3 = rand_vec(), rand_vec(), rand_vec()
        # bias: try to make maps invertible-ish by also trying structured ones
        res = evaluate(M1, M2, M3, t1, t2, t3)
        if res is not None:
            best = res
            break
        it += 1
        # also try a structured candidate periodically: identity-ish maps with distinct shifts
        if it % 4 == 0:
            Id = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            for (m1, m2, m3, tt1, tt2, tt3) in [
                (Id, Id, Id, rand_vec(), rand_vec(), rand_vec()),
                (rand_mat(), Id, Id, rand_vec(), rand_vec(), rand_vec()),
            ]:
                res = evaluate(m1, m2, m3, tt1, tt2, tt3)
                if res is not None:
                    best = res
                    break
            if best is not None:
                break
        if time.time() > _DEADLINE:
            break

    if best is None:
        # Fallback: return the seed's L-U-W style construction to guarantee validity.
        return _luw_fallback(k, g)

    return best


def _luw_fallback(k, g):
    """Fallback: Lazebnik-Ustimenko-Woldar construction (guaranteed valid, larger n)."""
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
