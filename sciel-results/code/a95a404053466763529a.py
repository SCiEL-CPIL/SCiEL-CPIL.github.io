"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def _prime_powers(limit):
    """All prime powers p^e <= limit, sorted."""
    out = set()
    # primes
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, limit + 1):
        if sieve[i]:
            out.add(i)
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    # prime powers
    for p in list(out):
        v = p * p
        while v <= limit:
            out.add(v)
            v *= p
    return sorted(out)


def _gf_tables(q):
    """Return (add, sub, mul) for GF(q), q prime or 4."""
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: (x * y) % q
    return add, sub, mul


def _luw_rule(d):
    """L-U-W rule: rule[c] = (a,b) for c in 2..d, odd d >= 3."""
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    return rule


def _luw_component(q, d):
    """Return (n, edges, adj) for the connected component of the zero point
    of the L-U-W graph over GF(q) with parameter d (odd)."""
    add, sub, mul = _gf_tables(q)
    rule = _luw_rule(d)

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
    index, order = {start: 0}, [start]
    edges = []
    adj = [[] for _ in range(len(order))]  # placeholder; resized below
    for vertex in order:
        vi = index[vertex]
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((vi, index[other]))
    n = len(order)
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    return n, edges, adj


def _greedy_trim(n, edges, k):
    """Remove edges so every vertex has degree exactly k. Returns list of
    kept edges or None if impossible."""
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    deg = [len(a) for a in adj]
    if any(dg < k for dg in deg):
        return None
    removed = set()
    # process vertices with surplus, remove edges to still-surplus neighbours
    changed = True
    while changed:
        changed = False
        for v in range(n):
            if deg[v] <= k:
                continue
            for u in adj[v]:
                if deg[v] <= k:
                    break
                if (v, u) in removed or (u, v) in removed:
                    continue
                if deg[u] > k:
                    removed.add((v, u))
                    deg[v] -= 1
                    deg[u] -= 1
                    changed = True
    if any(dg != k for dg in deg):
        return None
    kept = []
    for a, b in edges:
        if (a, b) not in removed and (b, a) not in removed:
            kept.append((a, b))
    return kept


def _verify_girth(n, edges, g):
    """Return True iff the graph is simple, k-regular (deg check done by caller),
    and has no cycle shorter than g. Uses non-backtracking walks from every vertex,
    depth up to g-1 (a cycle of length L appears as an NB-walk of length L)."""
    adj = [[] for _ in range(n)]
    edge_set = set()
    for a, b in edges:
        if a == b:
            return False
        if a > b:
            a, b = b, a
        if (a, b) in edge_set:
            return False
        edge_set.add((a, b))
        adj[a].append(b)
        adj[b].append(a)
    for v in range(n):
        # BFS over non-backtracking walks: state (node, prev), depth
        # A cycle of length L < g means some vertex is reached at depth L
        # by an NB-walk whose start != end... actually NB-walks from v: a
        # cycle through v of length L returns to v at depth L.
        seen_depth = {}  # (node, prev) -> min depth
        queue = []
        for u in adj[v]:
            queue.append((u, v, 1))
        while queue:
            node, prev, depth = queue.pop(0)
            if node == v and depth < g:
                return False
            if depth >= g - 1:
                continue
            key = (node, prev)
            if key in seen_depth:
                continue
            seen_depth[key] = depth
            for w in adj[node]:
                if w != prev:
                    queue.append((w, node, depth + 1))
    return True


def construct(k, g):
    deadline = time.time() + 140.0  # safety margin under 150s kill
    powers = _prime_powers(200)
    # candidate (q, d): q >= k prime power, d odd, d+5 >= g, and q^d not absurd
    candidates = []
    for q in powers:
        if q < k:
            continue
        for d in range(3, 40, 2):
            if d + 5 < g:
                continue
            # estimate size q^d; skip if too large to build quickly
            if q ** d > 5_000_000:
                break
            candidates.append((q, d))
    # order by estimated component size (q^d) ascending
    candidates.sort(key=lambda t: t[0] ** t[1])

    best = None
    for q, d in candidates:
        if time.time() > deadline:
            break
        try:
            n, edges, adj = _luw_component(q, d)
        except Exception:
            continue
        if n < k:
            continue
        kept = _greedy_trim(n, edges, k)
        if kept is None:
            continue
        if not _verify_girth(n, kept, g):
            continue
        if best is None or n < best[0]:
            best = (n, kept)
            if time.time() > deadline:
                break
    if best is None:
        # fallback: original q=k construction
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        n, edges, adj = _luw_component(q, d)
        return n, edges
    return best
# EVOLVE-BLOCK-END
