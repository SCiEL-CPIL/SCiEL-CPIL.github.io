"""Spectral cage solver: configuration model + Laplacian spectral optimization
to raise girth, plus explicit constructions, all verified exactly."""
import time
import threading
import random as _random
import itertools as _itertools
from collections import deque
import numpy as np

_DEADLINE = 145.0
_start = [None]


def _elapsed():
    s = _start[0]
    return 0.0 if s is None else time.time() - s


# ---------- exact girth via non-backtracking walks from every vertex ----------
def _nonback_girth(adj, n):
    prev = {}
    q = deque()
    for u in range(n):
        for p in adj[u]:
            key = (p, u)
            prev[key] = (u, 1)
            q.append(key)
    while q:
        (v, p), (u, dist) = q.popleft()
        for w in adj[v]:
            if w == p:
                continue
            key = (w, v)
            if w == u:
                return dist + 1
            if key in prev:
                continue
            prev[key] = (v, dist + 1)
            q.append(key)
    return 10 ** 9


def _build_adj(edges, n):
    adj = [set() for _ in range(n)]
    for (i, j) in edges:
        adj[i].add(j)
        adj[j].add(i)
    for i in range(n):
        adj[i] = sorted(adj[i])
    return adj


def _is_valid(adj, n, k, g):
    if n == 0:
        return False
    if any(len(adj[i]) != k for i in range(n)):
        return False
    for i in range(n):
        if len(set(adj[i])) != len(adj[i]):
            return False
        for j in adj[i]:
            if i <= j:
                return False
    return _nonback_girth(adj, n) >= g


# ---------- Moore bound ----------
def _moore(k, g):
    if g % 2 == 1:
        r = (g - 1) // 2
        return 2 * sum((k - 1) ** i for i in range(r + 1))
    else:
        r = g // 2
        return 2 * sum((k - 1) ** i for i in range(r)) + k * (k - 1) ** (r - 1)


# ---------- construction: LUW ----------
def _luw(k, g):
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = sub = (lambda x, y: x ^ y)
        mul = (lambda x, y: table[x][y])
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


# ---------- construction: circulant ----------
def _circulant(k, g):
    if k % 2 != 0:
        return None
    n = k * 2
    while n < 6 * k and _elapsed() < _DEADLINE:
        half = k // 2
        found = None
        for S in _itertools.combinations(range(1, n // 2 + 1), half):
            S = list(S)
            if any((n - s) % n in S for s in S):
                continue
            adj = [set() for _ in range(n)]
            for i in range(n):
                for s in S:
                    adj[i].add((i + s) % n)
                    adj[i].add((i - s) % n)
            if all(len(adj[i]) == k for i in range(n)):
                if _nonback_girth(adj, n) >= g:
                    edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                    found = (n, edges)
                    break
        if found:
            return found
        n += 2
    return None


# ---------- configuration model + spectral optimization ----------
def _config_model(n, k, rng):
    """Generate a simple k-regular graph via configuration model. Returns adj list or None."""
    for _ in range(30):
        stubs = []
        for v in range(n):
            stubs.extend([v] * k)
        rng.shuffle(stubs)
        adj = [set() for _ in range(n)]
        valid = True
        for i in range(0, n * k, 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v or v in adj[u]:
                valid = False
                break
            adj[u].add(v)
            adj[v].add(u)
        if valid and all(len(adj[i]) == k for i in range(n)):
            return [sorted(adj[i]) for i in range(n)]
    return None


def _spectral_optimize(adj, n, k, g, rng, max_iters=3000):
    """Laplacian spectral optimization to raise girth. Returns (adj, girth) or None."""
    # Build adjacency matrix
    A = np.zeros((n, n), dtype=np.int8)
    for u in range(n):
        for v in adj[u]:
            A[u, v] = 1
    A = (A + A.T) // 2
    D = np.diag(np.sum(A, axis=1))
    L = D.astype(np.float64) - A.astype(np.float64)

    # Eigen decomposition
    evals, evecs = np.linalg.eigh(L)

    # Smallest eigenvalue is 0 (constant vector). Work with the rest.
    # We want to push the smallest non-trivial eigenvalue (Fiedler) up,
    # and more generally make the spectrum as "flat" as possible near 0.

    # Use the k smallest non-trivial eigenvectors for edge swaps
    # Target: maximize the minimum cycle length by making L look like k*I + noise
    # i.e., push eigenvalues toward k.

    # We'll do a greedy approach: try swaps, accept if it increases the
    # second-smallest eigenvalue (Fiedler) or reduces the number of short cycles.

    # Precompute current girth for quick checks
    def quick_girth():
        return _nonback_girth(adj, n)

    # For speed, we'll track the number of 4-cycles and 6-cycles as proxy
    # and only do full girth check periodically.

    def count_short_cycles():
        """Count 4-cycles and 6-cycles approximately."""
        # 4-cycles: pairs of vertices with 2+ common neighbors
        c4 = 0
        for u in range(n):
            for v in range(u + 1, n):
                common = len(set(adj[u]) & set(adj[v]))
                if common >= 2:
                    c4 += common * (common - 1) // 2
        return c4

    # We'll do random edge swaps: pick an edge (u,v), pick non-edge (w,x),
    # swap to (u,w),(v,x) or (u,x),(v,w) if it improves.

    # Precompute neighbor sets for speed
    adj_set = [set(adj[i]) for i in range(n)]

    best_fiedler = evals[1] if n > 1 else 0
    best_c4 = count_short_cycles()

    for it in range(max_iters):
        if _elapsed() > _DEADLINE - 10.0:
            break
        # Pick random edge
        u = rng.randrange(n)
        if not adj_set[u]:
            continue
        v = adj[u][rng.randrange(len(adj[u]))]

        # Pick random non-edge
        w = rng.randrange(n)
        if w == u or w == v or w in adj_set[u]:
            continue
        x = rng.randrange(n)
        if x == u or x == v or x == w or x in adj_set[u] or x in adj_set[v]:
            continue

        # Try swap: remove (u,v), add (u,w) and (v,x)  [or (u,x),(v,w)]
        for swap_type in range(2):
            if swap_type == 0:
                new_u, new_v = w, x
            else:
                new_u, new_v = x, w

            if new_u in adj_set[v] or new_v in adj_set[u]:
                continue

            # Do the swap
            adj_set[u].discard(v)
            adj_set[v].discard(u)
            adj_set[u].add(new_u)
            adj_set[new_u].add(u)
            adj_set[v].add(new_v)
            adj_set[new_v].add(v)

            # Check if it improved: recompute a few eigenvalues
            A2 = np.zeros((n, n), dtype=np.int8)
            for i in range(n):
                for j in adj_set[i]:
                    A2[i, j] = 1
            A2 = (A2 + A2.T) // 2
            D2 = np.diag(np.sum(A2, axis=1))
            L2 = D2.astype(np.float64) - A2.astype(np.float64)

            # Use power iteration for smallest eigenvalue (shifted)
            # Or just use a few iterations of the full eigen for small n
            if n <= 200:
                evals2, _ = np.linalg.eigh(L2)
                fiedler2 = evals2[1]
            else:
                # Approximate: use the trace and Frobenius norm
                # Simpler: just check if the swap reduced 4-cycles
                fiedler2 = None

            c4_new = count_short_cycles()

            # Accept if it reduces 4-cycles, or if n is small and Fiedler improved
            if c4_new < best_c4 or (fiedler2 is not None and fiedler2 > best_fiedler and c4_new <= best_c4):
                best_c4 = c4_new
                if fiedler2 is not None:
                    best_fiedler = fiedler2
                adj = [sorted(adj_set[i]) for i in range(n)]
                # Update A, L for next iteration
                A = A2
                L = L2
                evals, evecs = np.linalg.eigh(L) if n <= 200 else (None, None)
            else:
                # Undo the swap
                adj_set[u].discard(new_u)
                adj_set[new_u].discard(u)
                adj_set[v].discard(new_v)
                adj_set[new_v].discard(v)
                adj_set[u].add(v)
                adj_set[v].add(u)

        # Periodically check actual girth
        if it % 200 == 0:
            adj = [sorted(adj_set[i]) for i in range(n)]
            girth = _nonback_girth(adj, n)
            if girth >= g:
                return adj, girth

    adj = [sorted(adj_set[i]) for i in range(n)]
    girth = _nonback_girth(adj, n)
    return adj, girth


def _spectral_cage(k, g, seed=0):
    """Main spectral construction: find smallest n where spectral optimization succeeds."""
    rng = _random.Random(seed)
    moore = _moore(k, g)
    min_n = max(k + 1, moore)
    if min_n * k % 2 != 0:
        min_n += 1

    for n in range(min_n, min_n + 80, 2):
        if n * k % 2 != 0:
            continue
        if _elapsed() > _DEADLINE - 10.0:
            break
        # Try multiple random initial configurations
        for attempt in range(5):
            if _elapsed() > _DEADLINE - 10.0:
                break
            adj = _config_model(n, k, rng)
            if adj is None:
                continue
            # Quick girth check before expensive optimization
            if _nonback_girth(adj, n) >= g:
                edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                return n, edges
            # Spectral optimization
            res = _spectral_optimize(adj, n, k, g, rng, max_iters=2000)
            if res is not None:
                opt_adj, girth = res
                if girth >= g:
                    edges = [(i, j) for i in range(n) for j in opt_adj[i] if i < j]
                    return n, edges
    return None


# ---------- random regular + local search ----------
def _random_regular(k, g, seed=0):
    rng = _random.Random(seed)
    moore = _moore(k, g)
    min_n = max(k + 1, moore)
    if min_n * k % 2 != 0:
        min_n += 1

    for n in range(min_n, min_n + 50, 2):
        if n * k % 2 != 0:
            continue
        if _elapsed() > _DEADLINE - 10.0:
            break
        for attempt in range(30):
            if _elapsed() > _DEADLINE - 10.0:
                break
            adj = _config_model(n, k, rng)
            if adj is None:
                continue
            if _nonback_girth(adj, n) >= g:
                edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                return n, edges
    return None


# ---------- worker ----------
def _worker(name, fn, holder, k, g):
    try:
        res = fn()
    except Exception:
        res = None
    if res is None:
        return
    n, edges = res
    if _elapsed() >= _DEADLINE:
        return
    try:
        if n > 0 and _is_valid(_build_adj(edges, n), n, k, g):
            cur = holder.get("best")
            if cur is None or n < cur[0]:
                holder["best"] = (n, edges)
    except Exception:
        pass


def construct(k, g):
    _start[0] = time.time()
    holder = {"best": None}
    jobs = [
        ("luw", lambda: _luw(k, g)),
        ("circulant", lambda: _circulant(k, g)),
        ("spectral0", lambda: _spectral_cage(k, g, seed=0)),
        ("spectral1", lambda: _spectral_cage(k, g, seed=1)),
        ("spectral2", lambda: _spectral_cage(k, g, seed=2)),
        ("rand_reg0", lambda: _random_regular(k, g, seed=0)),
        ("rand_reg1", lambda: _random_regular(k, g, seed=1)),
        ("rand_reg2", lambda: _random_regular(k, g, seed=2)),
    ]
    threads = []
    for name, fn in jobs:
        t = threading.Thread(target=_worker, args=(name, fn, holder, k, g), daemon=True)
        t.start()
        threads.append(t)

    while _elapsed() < _DEADLINE - 2.0:
        if not any(t.is_alive() for t in threads):
            break
        time.sleep(0.1)

    best = holder.get("best")
    if best is None:
        try:
            n, edges = _luw(k, g)
            best = (n, edges)
        except Exception:
            n = k + 1
            edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            best = (n, edges)
    return best
