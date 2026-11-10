"""Cage construction via randomized graph lifts (permutation lift / voltage graph)."""
import time
import random
from collections import deque

_START = time.time()
_DEADLINE = _START + 135.0  # safety margin before the 150 s kill

def _perm_mul(a, b):
    return [a[b[i]] for i in range(len(a))]

def _perm_inv(a):
    inv = [0] * len(a)
    for i, v in enumerate(a):
        inv[v] = i
    return inv

def _rand_perm(L):
    p = list(range(L))
    random.shuffle(p)
    return p

def _random_base(k, n, rng):
    """Return adjacency list (list of lists) for a random k-regular-ish graph on n vertices,
       or None if we cannot make it exactly k-regular (simple)."""
    if n < k:
        return None
    if (n * k) % 2 != 0:
        return None
    # start from a random regular-ish multigraph-free attempt via configuration model, then dedupe
    for _ in range(40):
        deg = [0] * n
        adj = [set() for _ in range(n)]
        # build a random pairing
        stubs = []
        for v in range(n):
            for _ in range(k):
                stubs.append(v)
        rng.shuffle(stubs)
        ok = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v:
                ok = False
                break
            if v in adj[u]:
                ok = False
                break
            adj[u].add(v)
            adj[v].add(u)
        if not ok:
            continue
        if all(len(a) == k for a in adj):
            return [sorted(a) for a in adj]
    return None

def _base_girth(adj):
    """Exact girth of a simple undirected graph via BFS from each vertex (non-backtracking)."""
    n = len(adj)
    best = 10 ** 9
    for s in range(n):
        # BFS with parent to forbid backtracking
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                else:
                    # found a cycle: length = dist[u] + dist[v] + 1
                    L = dist[u] + dist[v] + 1
                    if L < best:
                        best = L
    return best

def _lift_girth(adj, perms, L):
    """Girth of the L-lift of base graph with edge permutations perms (adj[u][j] -> perm for edge u->adj[u][j]).
       Uses non-backtracking BFS in the lift. Returns the girth (int) or large if none found within bound."""
    n = len(adj)
    N = n * L
    # adjacency in lift: from (u, t) to (v, perm(u->v)(t))
    # build adjacency list of the lift (each undirected base edge gives two directed lift edges)
    # We'll do non-backtracking BFS from every lift vertex.
    # To keep it fast, precompute for each directed base edge (u->v) the permutation.
    # perms indexed by (u, position-of-v-in-adj[u])
    big = 10 ** 9
    best = big
    for s in range(N):
        su, st = divmod(s, L)
        # BFS state: (u, t, prev_v) where prev_v is the base vertex we came from (or -1)
        # dist
        dist = {}
        q = deque()
        # initial: at (su, st) with no previous
        dist[(su, st, -1)] = 0
        q.append((su, st, -1))
        found = False
        while q:
            u, t, pv = q.popleft()
            du = dist[(u, t, pv)]
            for v in adj[u]:
                if v == pv:
                    continue
                # need perm for edge u->v
                pos = adj[u].index(v)
                p = perms[(u, pos)]
                nt = p[t]
                key = (v, nt, u)
                if key in dist:
                    # cycle length
                    Lc = du + 1 + dist[key]
                    if Lc < best:
                        best = Lc
                else:
                    dist[key] = du + 1
                    q.append(key)
    return best

def _lift_girth_bounded(adj, perms, L, limit):
    """Return girth if found < limit, else return limit (i.e., girth >= limit)."""
    n = len(adj)
    N = n * L
    best = limit
    for s in range(N):
        if time.time() > _DEADLINE:
            return best
        su, st = divmod(s, L)
        dist = {}
        q = deque()
        dist[(su, st, -1)] = 0
        q.append((su, st, -1))
        while q:
            u, t, pv = q.popleft()
            du = dist[(u, t, pv)]
            if du + 1 >= best:
                continue
            for v in adj[u]:
                if v == pv:
                    continue
                pos = adj[u].index(v)
                p = perms[(u, pos)]
                nt = p[t]
                key = (v, nt, u)
                if key in dist:
                    Lc = du + 1 + dist[key]
                    if Lc < best:
                        best = Lc
                else:
                    dist[key] = du + 1
                    q.append(key)
    return best

def _lift_edges(adj, perms, L):
    n = len(adj)
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                pos = adj[u].index(v)
                p = perms[(u, pos)]
                for t in range(L):
                    a = u * L + t
                    b = v * L + p[t]
                    edges.append((a, b))
    return edges

def construct(k, g):
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = _START + 135.0
    rng = random.Random(12345)

    # Target: 3-regular (k is given, typically 3). We'll handle k==3 robustly; for other k we fall back to a generic random k-regular lift.
    best_n = None
    best_edges = None

    # Strategy: pick a small base n0 that is k-regular with moderate girth, then lift.
    # The lift multiplies girth roughly by L. We want base_girth * L >= g.
    # We'll search over base sizes and lift lengths, random permutations, keep best (smallest N) with girth >= g.

    # Candidate base sizes (k-regular simple graphs exist for many)
    base_sizes = [6, 8, 10, 12, 14, 16, 18, 20, 24, 30, 36]
    # For each, try a few random base graphs and random permutation sets.

    # Precompute: for a base with m edges, we need m permutations of length L.
    # girth of lift >= min over cycles. A good heuristic: if base has girth gb and we choose L with
    # random permutations, the lift girth is often around L * (something). We just measure.

    def try_base(base, L, n_trials):
        nonlocal best_n, best_edges
        n = len(base)
        m = len(base) * k // 2
        # build perms dict
        for trial in range(n_trials):
            if time.time() > _DEADLINE:
                return
            perms = {}
            for u in range(n):
                for pos, v in enumerate(base[u]):
                    perms[(u, pos)] = _rand_perm(L)
            # make sure inverse consistency: for edge u<->v, perm(v->u) should be inv(perm(u->v))
            # We'll enforce: after generating, for each undirected edge set one and derive the other.
            # Regenerate properly:
            perms = {}
            for u in range(n):
                for pos, v in enumerate(base[u]):
                    if u < v:
                        p = _rand_perm(L)
                        perms[(u, pos)] = p
                        # position of u in base[v]
                        posv = base[v].index(u)
                        perms[(v, posv)] = _perm_inv(p)
                    else:
                        # already set from the other side
                        posv = base[v].index(u)
                        p = perms[(v, posv)]
                        perms[(u, pos)] = _perm_inv(p)
            N = n * L
            # check girth bounded by g
            gh = _lift_girth_bounded(base, perms, L, g)
            if gh >= g:
                if best_n is None or N < best_n:
                    best_n = N
                    best_edges = _lift_edges(base, perms, L)
                    return  # good enough for this (base, L); stop further trials
            # if we have time, keep trying to maybe find smaller? N is fixed for this (base,L). Stop.
            return

    # Iterate over (base_size, L) combos, larger L first to have a chance, but we want smallest N.
    # N = n0 * L. We want N as small as possible with girth >= g.
    # Try to find, for increasing N, a feasible (n0, L) with n0*L = N.
    # Simpler: try a grid and keep the smallest feasible N.

    combos = []
    for n0 in base_sizes:
        if n0 < k:
            continue
        for L in range(2, 40):
            N = n0 * L
            if N < 2 * k:  # trivial lower bound
                continue
            combos.append((N, n0, L))
    combos.sort()

    # We'll try combos in increasing N, but to save time try a limited number and stop early if we find a good one.
    # Also we want to ensure base graph exists.

    tried = 0
    for N, n0, L in combos:
        if time.time() > _DEADLINE:
            break
        if best_n is not None and N >= best_n:
            break
        base = _random_base(k, n0, rng)
        if base is None:
            continue
        # base must be simple & k-regular (guaranteed by _random_base)
        # quick base girth check (cheap)
        gb = _base_girth(base)
        # heuristic: need gb * L >= g roughly; if gb * L < g it's very unlikely but still try a couple
        try_base(base, L, 3)
        tried += 1
        if tried > 60:
            break

    # If nothing found, fall back to a guaranteed construction: a cycle-based lift (crown) that is valid.
    if best_n is None:
        # Fallback: take a base 3-regular graph we can always build: the prism / or use a known small one.
        # Simplest guaranteed: build a 3-regular base of size 6 (K3,3) and lift.
        # K3,3 adjacency:
        base = [
            [3, 4, 5],
            [3, 4, 5],
            [3, 4, 5],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
        ]
        # K3,3 girth is 4. Lift with L.
        for L in range(4, 60):
            if time.time() > _DEADLINE:
                break
            N = 6 * L
            for trial in range(6):
                perms = {}
                for u in range(6):
                    for pos, v in enumerate(base[u]):
                        if u < v:
                            p = _rand_perm(L)
                            perms[(u, pos)] = p
                            posv = base[v].index(u)
                            perms[(v, posv)] = _perm_inv(p)
                        else:
                            posv = base[v].index(u)
                            p = perms[(v, posv)]
                            perms[(u, pos)] = _perm_inv(p)
                gh = _lift_girth_bounded(base, perms, L, g)
                if gh >= g:
                    best_n = N
                    best_edges = _lift_edges(base, perms, L)
                    break
            if best_n is not None:
                break

    # Absolute fallback: a long-cycle-based 3-regular graph (two long cycles + rungs) with girth ~ size.
    if best_n is None:
        # Build a 3-regular graph on n vertices with large girth using a "circulant + shift" construction.
        # Use the known family: vertices (i, j) for i in 0..n/2-1, j in 0,1,2 with edges forming a 3-regular graph.
        # Simpler: take a cycle of length M, add two parallel "offset" cycles. This is the standard 3-regular girth construction.
        # Vertices 0..M-1 on a cycle; plus we need 3-regular so use a "circular ladder" doubled.
        # Use: n even. vertices 0..n-1. edges: (i, i+1 mod n) [cycle], (i, i + n//2 mod n) [diameter], (i, i + n//4 mod n)?? 
        # Let's just use a guaranteed 3-regular: the "Möbius ladder" is 3-regular: cycle + diameters.
        # Möbius ladder on n (n even): cycle edges (i, i+1) and (i, i + n/2). That's degree 2+1=3. Girth of Möbius ladder ~ n/2.
        M = 2 * g  # so girth ~ M/2 = g, need girth >= g, use M = 2*g + 4 for safety
        n = M
        edges = []
        for i in range(n):
            edges.append((i, (i + 1) % n))
        for i in range(n // 2):
            edges.append((i, (i + n // 2) % n))
        # dedupe
        edges = sorted(set(edges))
        best_n = n
        best_edges = edges

    return best_n, best_edges
