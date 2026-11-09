"""C6-free graph on 40 vertices: maximize edges via greedy build + local search."""
import time
import random
import numpy as np

N = 40
ALL_PAIRS = [(i, j) for i in range(N) for j in range(i + 1, N)]


def _init_adj():
    return [set() for _ in range(N)]


def _add(adj, u, v):
    adj[u].add(v)
    adj[v].add(u)


def _remove(adj, u, v):
    adj[u].discard(v)
    adj[v].discard(u)


def _has_c6(adj, u, v):
    """Return True if adding edge (u,v) creates a C6."""
    nu = adj[u]
    nv = adj[v]
    if len(nu) < 2 or len(nv) < 2:
        return False
    # neighbors of u
    a1, a2 = [], []
    for x in nu:
        (a1 if x < v else a2).append(x)
    # neighbors of v
    b1, b2 = [], []
    for y in nv:
        (b1 if y < u else b2).append(y)

    # Check pair (a in N(u), b in N(v)): need a,b,u,v distinct and a-b non-edge.
    # Then need a path of length 3 between a and b avoiding u,v (and a,b endpoints).
    # Path: a - x - y - b, x,y distinct, x,y not in {u,v,a,b}, a-x, x-y, y-b edges.

    # For each a in N(u):
    #   for each b in N(v), b != a, b not in N(u) (non-edge a-b):
    #       need x in N(a) \ {u,v,b}, y in N(b) \ {u,v,a} with x-y edge, x!=y
    # We can precompute per (a,b) but that's heavy. Instead iterate.

    for a in a1:
        na = adj[a]
        for b in b1:
            if b == a or b in nu:
                continue
            # need a-x-y-b
            nb = adj[b]
            # candidates x in na \ {u,v,b}
            # candidates y in nb \ {u,v,a}
            # check any x-y edge
            # small sets; do set intersection via neighbor sets
            # For each x, check if any y in nb with x-y edge
            for x in na:
                if x == u or x == v or x == b:
                    continue
                nx = adj[x]
                # need y in nb ∩ nx, y not in {u,v,a}
                for y in nx:
                    if y == u or y == v or y == a:
                        continue
                    if y in nb:
                        return True
    # symmetric role: also a in N(u) could be > u side? We partitioned N(u) into <v and >v.
    # The C6 is u-a-x-y-b-v-u. a in N(u), b in N(v). We covered a in a1 (<v). 
    # But a could be > v? a is a neighbor of u; its value relative to v doesn't matter for the cycle.
    # We must consider ALL a in N(u) and ALL b in N(v). Let's redo without partition.
    return _has_c6_full(adj, u, v)


def _has_c6_full(adj, u, v):
    nu = adj[u]
    nv = adj[v]
    if len(nu) < 2 or len(nv) < 2:
        return False
    for a in nu:
        if a == v:
            continue
        na = adj[a]
        for b in nv:
            if b == a or b == u:
                continue
            if b in nu:
                continue  # a-b edge -> not a valid C6 (would create smaller cycle)
            nb = adj[b]
            for x in na:
                if x == u or x == v or x == b:
                    continue
                nx = adj[x]
                for y in nx:
                    if y == u or y == v or y == a:
                        continue
                    if y in nb:
                        return True
    return False


def _edge_count(adj):
    return sum(len(s) for s in adj) // 2


def _greedy_build(rng, seed):
    adj = _init_adj()
    # Start from a base: random spanning structure + greedy add
    # Try to add edges in a randomized order
    order = list(ALL_PAIRS)
    rng.shuffle(order)
    for (u, v) in order:
        if v in adj[u]:
            continue
        if not _has_c6_full(adj, u, v):
            _add(adj, u, v)
    return adj


def _local_search(adj, rng, max_iter, time_limit):
    best = adj
    best_count = _edge_count(adj)
    start = time.time()
    cur = [set(s) for s in adj]
    cur_count = best_count
    for it in range(max_iter):
        if time.time() - start > time_limit:
            break
        # Try a random edge swap: remove (a,b), add (c,d) if C6-free and improves/keeps
        improved = False
        # pick a random existing edge to remove
        edges = []
        for u in range(N):
            for v in cur[u]:
                if u < v:
                    edges.append((u, v))
        if not edges:
            break
        a, b = rng.choice(edges)
        # pick a random non-edge to add
        # sample a few candidates
        for _ in range(200):
            c = rng.randrange(N)
            d = rng.randrange(N)
            if c == d or c >= d:
                continue
            if d in cur[c]:
                continue
            # remove a-b, add c-d
            _remove(cur, a, b)
            ok = not _has_c6_full(cur, c, d)
            if ok:
                _add(cur, c, d)
                new_count = _edge_count(cur)
                if new_count >= cur_count:
                    cur_count = new_count
                    if new_count > best_count:
                        best = [set(s) for s in cur]
                        best_count = new_count
                    improved = True
                    break
                else:
                    _remove(cur, c, d)
            else:
                _remove(cur, c, d)
            _add(cur, a, b)
        if not improved:
            _add(cur, a, b)
    return best, best_count


def _flip_search(adj, rng, max_iter, time_limit):
    """Try single edge flips: remove one edge, add another (net 0), keep if C6-free; 
       occasionally accept to explore. Then try to add edges greedily."""
    cur = [set(s) for s in adj]
    start = time.time()
    for it in range(max_iter):
        if time.time() - start > time_limit:
            break
        # pick random edge to remove
        edges = []
        for u in range(N):
            for v in cur[u]:
                if u < v:
                    edges.append((u, v))
        if not edges:
            break
        a, b = rng.choice(edges)
        _remove(cur, a, b)
        # try to add a random non-edge
        for _ in range(100):
            c = rng.randrange(N)
            d = rng.randrange(N)
            if c == d or c >= d:
                continue
            if d in cur[c]:
                continue
            if not _has_c6_full(cur, c, d):
                _add(cur, c, d)
                break
        else:
            _add(cur, a, b)
    return cur


def construct():
    deadline = 120.0
    start = time.time()
    rng = random.Random(12345)
    np.random.seed(12345)

    best_adj = None
    best_count = -1

    # Phase 1: multiple greedy builds with different orderings
    n_greedy = 12
    for i in range(n_greedy):
        if time.time() > deadline - 40:
            break
        adj = _greedy_build(rng, i)
        cnt = _edge_count(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = [set(s) for s in adj]

    # Phase 2: local search from best
    if best_adj is not None:
        remaining = deadline - 20 - (time.time() - start)
        if remaining > 5:
            adj2, cnt2 = _local_search(best_adj, rng, max_iter=20000, time_limit=remaining * 0.5)
            if cnt2 > best_count:
                best_count = cnt2
                best_adj = adj2

    # Phase 3: more greedy + local search
    for i in range(8):
        if time.time() > deadline - 25:
            break
        adj = _greedy_build(rng, 100 + i)
        cnt = _edge_count(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = [set(s) for s in adj]
        remaining = deadline - 20 - (time.time() - start)
        if remaining > 5:
            adj2, cnt2 = _local_search(adj, rng, max_iter=5000, time_limit=min(remaining * 0.3, 8))
            if cnt2 > best_count:
                best_count = cnt2
                best_adj = adj2

    # Phase 4: final local search on best
    if best_adj is not None:
        remaining = deadline - 15 - (time.time() - start)
        if remaining > 3:
            adj2, cnt2 = _local_search(best_adj, rng, max_iter=50000, time_limit=remaining * 0.6)
            if cnt2 > best_count:
                best_count = cnt2
                best_adj = adj2

    if best_adj is None:
        # fallback: 8 disjoint K5
        edges = []
        for k in range(8):
            edges += [[5 * k + a, 5 * k + b] for a in range(5) for b in range(a + 1, 5)]
        return edges

    edges = []
    for u in range(N):
        for v in best_adj[u]:
            if u < v:
                edges.append([u, v])
    return edges
