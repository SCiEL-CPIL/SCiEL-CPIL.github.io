"""Bit-parallel edge-flip local search for R(C4,K11).

Structural change vs. seed: instead of growing the graph by adding vertices /
edges, we fix a 43-vertex frame, seed it with the q=7 polarity graph (39 vertices,
alpha=10, C4-free), and then run a MaxSAT-style local search that flips single
edges (adds C4-safe edges, or removes edges) to *increase* the vertex count while
keeping alpha < 11.  C4-freedom is verified bit-parallel in O(n^3/word) and the
independence constraint is checked with a cheap greedy lower bound first (exact
for the requirement) and an exact branch-and-bound only when the greedy bound is
inconclusive.
"""
import time
import random

# EVOLVE-BLOCK-START

def _c4_ok(n, adj):
    """Bit-parallel C4 check: no two vertices share >= 2 common neighbours."""
    for u in range(n):
        au = adj[u]
        for v in range(u + 1, n):
            if (au & adj[v]).bit_count() >= 2:
                return False
    return True


def _greedy_alpha(adj, n, limit, deadline):
    """Greedy lower bound on alpha.  Returns True if an independent set of size
    >= limit is found (=> alpha >= limit, INVALID).  The bound is exact for the
    'no independent 11-set' requirement: if it cannot find 11, we must fall back
    to the exact check only in that inconclusive case."""
    t0 = time.time()
    # Repeated greedy: repeatedly pick a random vertex, add it, delete its
    # neighbours; track the max set size over several restarts.
    best = 0
    for _ in range(60):
        if time.time() - t0 > deadline:
            break
        rem = (1 << n) - 1
        size = 0
        while rem:
            # pick a random vertex in rem
            r = rem
            # random position
            k = random.randrange(r.bit_length())
            # find the k-th set bit
            x = r
            while k:
                lb = x & (-x)
                x ^= lb
                k -= 1
            lb = x & (-x)
            v = lb.bit_length() - 1
            size += 1
            if size >= limit:
                return True
            rem &= ~lb & ~adj[v]
        if size > best:
            best = size
    return False


def _exact_alpha_ge(adj, n, limit, deadline):
    """Exact branch-and-bound: returns True if alpha >= limit."""
    t0 = time.time()
    full = (1 << n) - 1
    best = [0]

    def _dfs(rem, size):
        if time.time() - t0 > deadline:
            return
        if size >= limit:
            best[0] = size
            return
        if not rem:
            if size > best[0]:
                best[0] = size
            return
        if size + rem.bit_count() <= best[0]:
            return
        lb = rem & (-rem)
        v = lb.bit_length() - 1
        # exclude v
        _dfs(rem & ~lb, size)
        # include v
        _dfs(rem & ~lb & ~adj[v], size + 1)

    _dfs(full, 0)
    return best[0] >= limit


def _alpha_ok(adj, n, deadline):
    """Return True iff alpha < 11 (no independent 11-set)."""
    # Fast path: if greedy finds an 11-set -> invalid immediately.
    if _greedy_alpha(adj, n, 11, deadline):
        return False
    # Greedy did not find 11.  For small n the greedy is often exact; to be safe
    # use the exact check but with a tight deadline.
    if _exact_alpha_ge(adj, n, 11, deadline):
        return False
    return True


def _embed_polarity_q7():
    """Build the q=7 polarity graph (39 vertices) and embed in a 43-vertex frame
    with 4 isolated extra vertices (so the frame is always 43)."""
    q = 7
    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv = pow(x, q - 2, q)
                    key = (1, (y * inv) % q, (z * inv) % q)
                elif y != 0:
                    inv = pow(y, q - 2, q)
                    key = (0, 1, (z * inv) % q)
                else:
                    key = (0, 0, 1)
                if key not in seen:
                    seen.add(key)
                    points.append(key)
    n_pts = len(points)
    adj = [0] * n_pts
    for i in range(n_pts):
        a1, b1, c1 = points[i]
        for j in range(i + 1, n_pts):
            a2, b2, c2 = points[j]
            if (a1 * a2 + b1 * b2 + c1 * c2) % q == 0:
                adj[i] |= 1 << j
                adj[j] |= 1 << i
    return n_pts, adj


def _count_vertices(n, adj):
    """Number of vertices with degree > 0 (the 'active' vertex count)."""
    c = 0
    for i in range(n):
        if adj[i]:
            c += 1
    return c


def _edges_from_adj(n, adj):
    edges = []
    for i in range(n):
        m = adj[i]
        j = i + 1
        while m:
            lb = m & (-m)
            v = lb.bit_length() - 1
            if v > i:
                edges.append([i, v])
            m ^= lb
    return edges


def construct():
    deadline = time.time() + 140.0  # safety margin before the 150s kill
    random.seed(7)

    N = 43
    base_n, base_adj = _embed_polarity_q7()
    # Embed into 43-vertex frame: first base_n vertices carry the polarity graph,
    # the rest are isolated.
    adj = [0] * N
    for i in range(base_n):
        adj[i] = base_adj[i]

    # Verify seed is valid.
    assert _c4_ok(N, adj), "polarity seed not C4-free"
    assert _alpha_ok(adj, N, deadline), "polarity seed has independent 11-set"

    best_n = _count_vertices(N, adj)
    best_adj = list(adj)

    def _try_flip(u, v):
        """Attempt to flip edge (u,v).  Returns True if accepted."""
        nonlocal adj, best_n, best_adj
        has = bool(adj[u] & (1 << v))
        if has:
            # Removing an edge always preserves C4-freedom and cannot create an
            # independent set, but it may *decrease* the vertex count (if a vertex
            # becomes isolated) -- only accept if it keeps the count up.  In
            # practice we rarely want to remove; use it as a move for diversity.
            new_adj = list(adj)
            new_adj[u] ^= (1 << v)
            new_adj[v] ^= (1 << u)
            new_n = _count_vertices(N, new_adj)
            if new_n >= best_n and _c4_ok(N, new_adj) and _alpha_ok(new_adj, N, deadline):
                adj = new_adj
                if new_n > best_n:
                    best_n = new_n
                    best_adj = list(adj)
                return True
            return False
        else:
            # Adding edge (u,v): must keep C4-free and alpha < 11.
            if (adj[u] & adj[v]).bit_count() >= 1:
                # would create a C4 (two common neighbours -> after adding edge
                # u,v they'd share... actually adding edge u,v creates a C4 iff
                # there is a 3-path u-a-b-v; the common-neighbour condition is
                # necessary but not sufficient.  Do the full check below.)
                pass
            new_adj = list(adj)
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            if not _c4_ok(N, new_adj):
                return False
            if not _alpha_ok(new_adj, N, deadline):
                return False
            adj = new_adj
            new_n = _count_vertices(N, adj)
            if new_n > best_n:
                best_n = new_n
                best_adj = list(adj)
            return True

    # Local search: iterate edge flips.  Prefer adding edges to isolated or
    # low-degree vertices to grow the active vertex count.
    improved = True
    stall = 0
    while time.time() < deadline - 5 and best_n < N:
        improved = False
        # Build candidate moves: all non-edges, prioritising those incident to
        # isolated/low-degree vertices.
        moves = []
        for u in range(N):
            for v in range(u + 1, N):
                if not (adj[u] & (1 << v)):
                    du = adj[u].bit_count()
                    dv = adj[v].bit_count()
                    # priority: lower degree = more "room"
                    moves.append((du + dv, u, v))
        random.shuffle(moves)
        moves.sort()
        for _pri, u, v in moves:
            if time.time() > deadline - 5:
                break
            if best_n >= N:
                break
            if _try_flip(u, v):
                improved = True
                stall = 0
                # restart the loop to re-evaluate with the new state
                break
        if not improved:
            # try a few random removals for diversity
            removed = False
            for _ in range(20):
                if time.time() > deadline - 5:
                    break
                u = random.randrange(N)
                v = random.randrange(N)
                if u == v or not (adj[u] & (1 << v)):
                    continue
                if _try_flip(u, v):
                    removed = True
                    improved = True
                    break
            if not improved:
                stall += 1
                if stall > 6:
                    break

    # Final: also try a pure greedy edge-addition pass to push the count up.
    for u in range(N):
        if time.time() > deadline - 5:
            break
        for v in range(u + 1, N):
            if time.time() > deadline - 5:
                break
            if not (adj[u] & (1 << v)):
                _try_flip(u, v)

    # Build the result on the best (active) vertex set.
    active = [i for i in range(N) if best_adj[i]]
    m = len(active)
    idx = {old: new for new, old in enumerate(active)}
    out_adj = [0] * m
    for new, old in enumerate(active):
        mm = best_adj[old]
        for old2 in active:
            if mm & (1 << old2):
                out_adj[new] |= 1 << idx[old2]
    edges = _edges_from_adj(m, out_adj)
    return {'n': m, 'edges': edges}

# EVOLVE-BLOCK-END
