import time
import itertools
import random

_DEADLINE = 150.0
_MARGIN = 8.0  # safety margin before the 150s kill

def _violation(adj, n):
    """Return a violating subset (size>=6 inducing a regular graph) or None."""
    # Precompute degrees
    deg = [0] * n
    for i in range(n):
        r = adj[i]
        d = 0
        for j in range(n):
            if r[j]:
                d += 1
        deg[i] = d

    # Check sizes from 6 upward; prefer larger subsets (more regularity candidates)
    for size in range(n, 5, -1):
        for combo in itertools.combinations(range(n), size):
            # induced degrees
            idx = list(combo)
            # quick: all induced degrees must be equal
            first = None
            ok = True
            for a in idx:
                d = 0
                r = adj[a]
                for b in idx:
                    if a != b and r[b]:
                        d += 1
                if first is None:
                    first = d
                elif d != first:
                    ok = False
                    break
            if ok and first is not None:
                return idx
    return None

def _violation_fast(adj, n):
    """Faster: only check sizes 6..min(n,10) first for speed, then larger if needed."""
    max_size = min(n, 12)
    for size in range(6, max_size + 1):
        for combo in itertools.combinations(range(n), size):
            idx = list(combo)
            first = None
            ok = True
            for a in idx:
                d = 0
                r = adj[a]
                for b in idx:
                    if a != b and r[b]:
                        d += 1
                if first is None:
                    first = d
                elif d != first:
                    ok = False
                    break
            if ok and first is not None:
                return idx
    # if none found in small sizes, check larger
    for size in range(max_size + 1, n + 1):
        for combo in itertools.combinations(range(n), size):
            idx = list(combo)
            first = None
            ok = True
            for a in idx:
                d = 0
                r = adj[a]
                for b in idx:
                    if a != b and r[b]:
                        d += 1
                if first is None:
                    first = d
                elif d != first:
                    ok = False
                    break
                if not ok:
                    break
            if ok and first is not None:
                return idx
    return None

def _is_valid(adj, n):
    """Check whether NO subset of size>=6 induces a regular subgraph."""
    return _violation(adj, n) is None

def _repair(adj, n, subset, rng, max_tries=200):
    """Try to flip one internal edge to break regularity of `subset`.
    Returns True if a flip made this subset non-regular (we re-check globally later)."""
    m = len(subset)
    # current induced degrees
    def induced_deg(v):
        d = 0
        r = adj[v]
        for b in subset:
            if v != b and r[b]:
                d += 1
        return d

    # all degrees equal
    target = induced_deg(subset[0])
    # candidate edges to flip: internal edges
    edges = []
    for i in range(m):
        for j in range(i + 1, m):
            edges.append((subset[i], subset[j]))
    rng.shuffle(edges)

    for (u, v) in edges[:max_tries]:
        # flip edge u-v
        adj[u][v] = not adj[u][v]
        adj[v][u] = adj[u][v]
        # check if this subset is now non-regular
        first = None
        reg = True
        for a in subset:
            d = 0
            r = adj[a]
            for b in subset:
                if a != b and r[b]:
                    d += 1
            if first is None:
                first = d
            elif d != first:
                reg = False
                break
        if not reg:
            # keep the flip
            return True
        else:
            # revert
            adj[u][v] = not adj[u][v]
            adj[v][u] = adj[u][v]
    return False

def _random_adj(n, p, rng):
    adj = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                adj[i][j] = True
                adj[j][i] = True
    return adj

def _edges_from_adj(adj, n):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                edges.append([i, j])
    return edges

def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE - _MARGIN
    rng = random.Random(12345)

    best_n = 0
    best_edges = []

    # Try sizes from 21 down to 6; stop early if we find valid at higher n
    # Start high to attempt the breakthrough, but fall back.
    for n in [21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6]:
        if time.time() > deadline:
            break
        found = False
        # multiple restarts at this size
        restarts = 30 if n <= 18 else 12
        for r in range(restarts):
            if time.time() > deadline:
                break
            # choose density
            p = 0.5
            adj = _random_adj(n, p, rng)
            # local search
            iters = 4000 if n <= 18 else 1500
            for it in range(iters):
                if time.time() > deadline:
                    break
                v = _violation_fast(adj, n)
                if v is None:
                    # valid!
                    if n > best_n:
                        best_n = n
                        best_edges = _edges_from_adj(adj, n)
                    found = True
                    break
                # try to repair
                ok = _repair(adj, n, v, rng)
                if not ok:
                    # perturbation: flip a random internal edge of v
                    m = len(v)
                    i = rng.randrange(m)
                    j = rng.randrange(m)
                    if i != j:
                        u, w = v[i], v[j]
                        adj[u][w] = not adj[u][w]
                        adj[w][u] = adj[u][w]
            if found:
                # keep searching a bit more for same n? We already have best_n=n.
                # Try to see if we can push further only via outer loop.
                break
        if found:
            # continue to try larger? larger already tried. So we can break.
            # But we iterate descending, so once found at n, smaller won't improve.
            break
        if time.time() > deadline:
            break

    # Fallback: if nothing found (shouldn't happen for n>=6 with enough time),
    # produce a trivial valid small graph.
    if best_n == 0:
        # empty graph on 5 vertices is trivially valid (no 6-subset)
        return {'n': 5, 'edges': []}

    return {'n': best_n, 'edges': best_edges}
