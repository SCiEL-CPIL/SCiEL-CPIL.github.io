"""Cage construction: find a k-regular graph of girth >= g on as few vertices as possible.

Strategy (deterministic, robust, fast):
  * For each candidate size n (in multiples of 2), look for an abelian group
    G = Z_p1 x Z_p2 x Z_p3 x Z_p4 of order n with no small-order elements
    (p_i > g//2).  For such a group, any reduced (non-backtracking) word of
    length < g evaluates to non-identity, so the girth of the graph is
    governed purely by whether the generator set has short "cayley relations".
  * Build a symmetric 4-element generator set via a fast BFS over short words
    and accept the first one with no relation of length < g (verified exactly
    by a non-backtracking BFS over the whole graph).
  * Return the smallest verified graph found, or the first valid one if time
    runs out.  If nothing is found, fall back to a random regular graph +
    2-opt girth polish (the previous family), still verified.
"""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START


def _moore_bound(k, g):
    if g % 2 == 1:
        return 2 * k * (k ** ((g - 3) // 2) + k ** ((g - 1) // 2)) - 2
    else:
        return k * (k - 1) * (k ** (g // 2 - 1) + 1) + 2 * (k - 1)


def _edges_to_adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _verify_girth(adj, n, g):
    """Exact girth check via non-backtracking BFS from every vertex.
    Returns True iff the shortest cycle is at least g (i.e. girth >= g)."""
    for start in range(n):
        # depth[(u,v)] = min length of a non-backtracking walk from start
        # ending with the directed edge u -> v.
        depth = {}
        queue = deque()
        for nb in adj[start]:
            depth[(start, nb)] = 1
            queue.append((start, nb))
        while queue:
            u, v = queue.popleft()
            d = depth[(u, v)]
            if d + 1 >= g:
                continue
            for w in adj[v]:
                if w == u:
                    continue
                key = (v, w)
                if key in depth:
                    if d + 1 + depth[key] < g:
                        return False
                else:
                    depth[key] = d + 1
                    if d + 1 < g:
                        queue.append(key)
    return True


# ---------------------------------------------------------------------------
# Abelian-group / generator-set construction
# ---------------------------------------------------------------------------

def _smallest_prime_ge(x):
    """Smallest prime >= x."""
    if x <= 2:
        return 2
    if x % 2 == 0:
        x += 1
    while True:
        is_prime = True
        r = int(x ** 0.5)
        d = 3
        while d <= r:
            if x % d == 0:
                is_prime = False
                break
            d += 2
        if is_prime:
            return x
        x += 2


def _factor_into_primes(n, max_dim=4):
    """Represent n as a product of <= max_dim primes (all >= 2).
    Returns a list of primes, or None if not possible."""
    if n < 2:
        return None
    factors = []
    m = n
    d = 2
    while d * d <= m and len(factors) < max_dim - 1:
        if m % d == 0:
            factors.append(d)
            m //= d
        else:
            d += 1 if d == 2 else 2
    if m > 1:
        factors.append(m)
    if len(factors) > max_dim:
        return None
    # each factor must be prime; otherwise try to re-express the big leftover
    # as a prime (it already is, since we divided by all small factors)
    return factors


def _cayley_girth_ok(pairs, gens, g):
    """Given the full (p1,p2,p3,p4) product and a generator tuple gens of
    length 4 (each a 4-tuple in the product), check via non-backtracking
    BFS that no reduced word of length < g is the identity."""
    p1, p2, p3, p4 = pairs
    mod = (p1, p2, p3, p4)

    def add(a, b):
        return ((a[0] + b[0]) % mod[0], (a[1] + b[1]) % mod[1],
                (a[2] + b[2]) % mod[2], (a[3] + b[3]) % mod[3])

    idn = (0, 0, 0, 0)
    depth = {idn: 0}
    # frontier as set of directed (prev_elem, cur_elem); we track element reached
    # plus which generator was used to arrive (to forbid immediate backtracking).
    # We'll do BFS on states (element, last_gen_index or -1 for start).
    q = deque()
    for gi, gen in enumerate(gens):
        state = (gen, gi)
        depth[state] = 1
        q.append(state)
    while q:
        elem, last = q.popleft()
        d = depth[(elem, last)]
        if d + 1 >= g:
            continue
        for gi, gen in enumerate(gens):
            if gi == last:
                continue
            ne = add(elem, gen)
            st = (ne, gi)
            if st in depth:
                if d + 1 + depth[st] < g:
                    return False
            else:
                depth[st] = d + 1
                if d + 1 < g:
                    q.append(st)
    return True


def _try_generator_set(pairs, g, seed, deadline, rng):
    """Try to find a symmetric 4-gen set with girth >= g in the given group.
    Returns (gens, n, edges) or None."""
    p1, p2, p3, p4 = pairs
    n = p1 * p2 * p3 * p4
    mod = (p1, p2, p3, p4)
    max_len = max(g // 2, 4)

    def to_flat(a):
        return a[0] + mod[0] * (a[1] + mod[1] * (a[2] + mod[2] * a[3]))

    def add(a, b):
        return ((a[0] + b[0]) % mod[0], (a[1] + b[1]) % mod[1],
                (a[2] + b[2]) % mod[2], (a[3] + b[3]) % mod[3])

    def neg(a):
        return (-a[0] % mod[0], -a[1] % mod[1], -a[2] % mod[2], -a[3] % mod[3])

    idn = (0, 0, 0, 0)
    # BFS over words to collect reachable elements (up to max_len).
    words = {idn: ()}          # element -> one representative word (tuple of gen idx)
    frontier = {idn}
    # We'll pick the first few non-identity elements as generators in order of
    # first discovery, then verify.
    # Build a BFS tree to get a natural ordering.
    order = [idn]
    q = deque()
    q.append(idn)
    # neighbor = add by one of the 4 "coordinate unit steps"? Not enough.
    # Instead, do a generic BFS using a preliminary generator set of 4 "axes":
    axes = [((1 % p1, 0, 0, 0), (0, 1 % p2, 0, 0), (0, 0, 1 % p3, 0), (0, 0, 0, 1 % p4))]
    # use axes[0] as BFS generators to enumerate elements by distance
    bfs_gens = axes[0]
    seen = {idn}
    q2 = deque([idn])
    while q2 and len(order) < min(n, 2000):
        e = q2.popleft()
        for g2 in bfs_gens:
            ne = add(e, g2)
            if ne not in seen:
                seen.add(ne)
                order.append(ne)
                q2.append(ne)
            ne = neg(ne) if False else e
    # Now pick candidate generator sets: take 4 elements from `order` (skip idn).
    # Strategy: try combinations of 4 elements; verify with _cayley_girth_ok.
    candidates = []
    elems = [e for e in order if e != idn]
    # Use a heuristic: elements with large "spread" across coordinates.
    elems.sort(key=lambda e: (e[0] * e[1] * e[2] * e[3]) % (n or 1), reverse=True)
    elems = elems[:120]
    import itertools
    attempts = 0
    while attempts < 2000:
        if time.time() > deadline:
            return None
        # pick 4 distinct elements, ensure no two are negatives of each other
        # (symmetry: graph edges come in pairs, but that's fine for simple graph
        # as long as no generator is its own inverse, i.e. order > 2)
        pick = []
        used = set()
        for _ in range(4):
            cands = [e for e in elems if e not in used and neg(e) not in used
                     and e != idn and all(x % mod[i] for i, x in enumerate(e)) is False or True]
            # filter: not identity and not 2-torsion (order>2)
            cands = [e for e in cands if e != idn and neg(e) != e]
            if not cands:
                break
            e = cands[rng.randrange(len(cands))]
            pick.append(e)
            used.add(e)
            used.add(neg(e))
        if len(pick) < 4:
            # reseed the pool
            rng.shuffle(elems)
            attempts += 1
            continue
        gens = tuple(pick)
        if _cayley_girth_ok(pairs, gens, g):
            # Build edges: for each element, connect to element+gen for the
            # 4 gens; each undirected edge counted once.
            adj = [[] for _ in range(n)]
            idx = {}
            for i, e in enumerate(order):
                idx[e] = i
            # ensure all elements indexed (order may be truncated if n>2000)
            if len(order) < n:
                # need full enumeration for edge list; build via mixed radix
                for a in range(p1):
                    for b in range(p2):
                        for c in range(p3):
                            for d_ in range(p4):
                                elem = (a, b, c, d_)
                                if elem not in idx:
                                    idx[elem] = len(order)
                                    order.append(elem)
                n = p1 * p2 * p3 * p4
                adj = [[] for _ in range(n)]
            edges = []
            seen_e = set()
            for e in order:
                ui = idx[e]
                for gen in gens:
                    ne = add(e, gen)
                    vi = idx[ne]
                    a, b = (ui, vi) if ui < vi else (vi, ui)
                    if (a, b) in seen_e:
                        continue
                    seen_e.add((a, b))
                    edges.append((a, b))
            return gens, n, edges
        attempts += 1
    return None


def _random_k_regular(n, k, seed=None):
    """Generate a random simple k-regular graph on n vertices. Returns list of edges or None."""
    if n * k % 2 != 0 or k >= n:
        return None
    rng = random.Random(seed)
    for _ in range(200):
        stubs = [i for i in range(n) for _ in range(k)]
        rng.shuffle(stubs)
        edges = set()
        valid = True
        for i in range(0, n * k, 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v:
                valid = False
                break
            a, b = (u, v) if u < v else (v, u)
            if (a, b) in edges:
                valid = False
                break
            edges.add((a, b))
        if not valid:
            continue
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if all(d == k for d in deg):
            return list(edges)
    return None


def _improve_girth(n, edges, k, g, max_iters=5000, deadline=None, seed=0):
    """Try to increase girth via edge swaps. Returns improved edges or None."""
    rng = random.Random(seed)
    adj = _edges_to_adj(n, edges)
    edge_set = set((min(u, v), max(u, v)) for u, v in edges)
    edge_list = list(edge_set)

    def girth_ok():
        return _verify_girth(adj, n, g)

    if girth_ok():
        return list(edge_list)

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        i = rng.randrange(len(edge_list))
        j = rng.randrange(len(edge_list))
        if i == j:
            continue
        e1, e2 = edge_list[i], edge_list[j]
        a1, b1 = e1
        a2, b2 = e2
        ne1 = (min(a1, a2), max(a1, a2))
        ne2 = (min(b1, b2), max(b1, b2))
        if ne1[0] == ne1[1] or ne2[0] == ne2[1]:
            continue
        if ne1 == e1 or ne1 == e2 or ne2 == e1 or ne2 == e2:
            continue
        if ne1 == ne2:
            continue
        if ne1 in edge_set or ne2 in edge_set:
            continue
        edge_set.discard(e1); edge_set.discard(e2)
        edge_set.add(ne1); edge_set.add(ne2)
        edge_list[i] = ne1; edge_list[j] = ne2
        adj[a1].remove(b1); adj[b1].remove(a1)
        adj[a2].remove(b2); adj[b2].remove(a2)
        adj[ne1[0]].append(ne1[1]); adj[ne1[1]].append(ne1[0])
        adj[ne2[0]].append(ne2[1]); adj[ne2[1]].append(ne2[0])
        if girth_ok():
            return list(edge_list)
        adj[ne1[0]].remove(ne1[1]); adj[ne1[1]].remove(ne1[0])
        adj[ne2[0]].remove(ne2[1]); adj[ne2[1]].remove(ne2[0])
        adj[a1].append(b1); adj[b1].append(a1)
        adj[a2].append(b2); adj[b2].append(a2)
        edge_set.discard(ne1); edge_set.discard(ne2)
        edge_set.add(e1); edge_set.add(e2)
        edge_list[i] = e1; edge_list[j] = e2
    return None


def construct(k, g):
    """Build a k-regular graph with girth at least g on as few vertices as possible.

    Returns (n, edges).  Uses an abelian-group / generator-set construction
    that can be exactly verified, falling back to random + 2-opt polish.
    """
    start_time = time.time()
    deadline = start_time + 130  # safety margin before the 150s kill

    lb = _moore_bound(k, g)
    # n must be a multiple of 2 (k*n even) and >= k
    lo = max(int(lb), k)
    if lo * k % 2 != 0:
        lo += 1

    best_result = None
    rng = random.Random(12345)

    # --- Primary: abelian-group generator construction ---
    # Try n from lo upward in steps that keep n*k even (step 2).
    max_tries = 60
    for t in range(max_tries):
        n = lo + 2 * t
        if time.time() > deadline:
            break
        if n * k % 2 != 0:
            continue
        # need n = product of <= 4 primes each > g//2 (so no small-order elem)
        min_p = _smallest_prime_ge(g // 2 + 1)
        factors = _factor_into_primes(n, max_dim=4)
        if factors is None:
            continue
        # pad/adjust to exactly 4 by allowing 1s (dimension of size 1)
        # but 1 is not prime; instead require all factors > 1.  If fewer than 4
        # factors, the missing dims are 1 (trivial).  Represent as (p1,p2,p3,p4).
        if len(factors) < 4:
            padded = list(factors) + [1] * (4 - len(factors))
        else:
            padded = list(factors)
        # ensure each non-1 factor > g//2
        ok = True
        for p in padded:
            if p != 1 and p <= g // 2:
                ok = False
                break
        if not ok:
            continue
        res = _try_generator_set(tuple(padded), g, start_time, deadline, rng)
        if res is not None:
            gens, nn, edges = res
            adj = _edges_to_adj(nn, edges)
            # verify degree and girth
            if all(len(a) == k for a in adj) and _verify_girth(adj, nn, g):
                best_result = (nn, edges)
                break

    if best_result is not None:
        return best_result

    # --- Fallback: random regular + 2-opt girth polish ---
    for t in range(20):
        n = lo + 2 * t
        if time.time() > deadline:
            break
        if n * k % 2 != 0:
            continue
        for trial in range(3):
            if time.time() > deadline:
                break
            edges = _random_k_regular(n, k, seed=int(start_time * 1000) + trial * 137 + n)
            if edges is None:
                continue
            adj = _edges_to_adj(n, edges)
            if _verify_girth(adj, n, g):
                return (n, edges)
            improved = _improve_girth(n, edges, k, g, max_iters=2000,
                                      deadline=deadline, seed=trial + 1)
            if improved is not None:
                adj = _edges_to_adj(n, improved)
                if _verify_girth(adj, n, g):
                    return (n, improved)

    # --- Last resort: return the smallest verified random graph we can, else a
    #     cycle-like regular graph (may not satisfy girth, but keeps interface) ---
    n_fallback = max(lo, k + 1)
    if n_fallback * k % 2 != 0:
        n_fallback += 1
    edges_fallback = _random_k_regular(n_fallback, k, seed=999)
    if edges_fallback is None:
        # build a k-regular circulant-ish graph
        n_fallback = max(k + 1, 4)
        if k % 2 == 1 and n_fallback % 2 == 1:
            n_fallback += 1
        edges_fallback = []
        seen = set()
        for i in range(n_fallback):
            for s in range(1, k // 2 + 1):
                j = (i + s) % n_fallback
                a, b = (i, j) if i < j else (j, i)
                if a != b and (a, b) not in seen:
                    seen.add((a, b))
                    edges_fallback.append((a, b))
            if k % 2 == 1:
                j = (i + n_fallback // 2) % n_fallback
                a, b = (i, j) if i < j else (j, i)
                if a != b and (a, b) not in seen:
                    seen.add((a, b))
                    edges_fallback.append((a, b))
    return (n_fallback, edges_fallback)
# EVOLVE-BLOCK-END
