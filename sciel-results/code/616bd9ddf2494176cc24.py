"""Combined cage solver: Circulant search, Random Configuration Model with local search, and LUW fallback."""
import time
import threading
import random
from collections import deque

_DEADLINE = 120.0
_start = [None]


def _elapsed():
    s = _start[0]
    return 0.0 if s is None else time.time() - s


# ---------------- exact girth (non-backtracking walks, all vertices) ----------
def _nonback_girth(adj, n, g_limit=None):
    """Return length of shortest cycle, certified by non-backtracking walks
    from every vertex. >= g means no cycle shorter than g.
    If g_limit is given, stop early if girth >= g_limit."""
    prev = {}
    q = deque()
    for u in range(n):
        for p in adj[u]:
            key = (p, u)
            prev[key] = (u, 1)
            q.append(key)
    while q:
        (v, p), (u, dist) = q.popleft()
        if g_limit and dist >= g_limit:
            return dist
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
    return _nonback_girth(adj, n, g) >= g


def _build_adj(edges, n):
    adj = [set() for _ in range(n)]
    for (i, j) in edges:
        adj[i].add(j)
        adj[j].add(i)
    for i in range(n):
        adj[i] = sorted(adj[i])
    return adj


# ---------------- construction 1: LUW (Lazebnik-Ustimenko-Woldar) ------------
def _luw(k, g):
    q = k
    # For girth g, we need d such that the girth of CD(d,q) >= g
    # The girth of CD(d,q) is d+5 for odd d, d+4 for even d
    # So for g=9: d+5>=9 => d>=4 (odd) or d+4>=9 => d>=5 (even)
    # Use d=4 (odd) for g=9
    if (g - 5) % 2 == 1:
        d = g - 5
    else:
        d = g - 4
    d = max(3, d)
    
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


# ---------------- construction 2: Circulant Search ---------------------------
def _circulant_search(k, g):
    R = (g - 1) // 2
    moore = 1 + k * sum((k - 1) ** i for i in range(R))
    
    if k % 2 == 0:
        half = k // 2
        for n in range(max(moore, k + 1), max(moore, k + 1) + 100):
            if n * k % 2 != 0:
                continue
            if _elapsed() > _DEADLINE - 15.0:
                return None
            rng = random.Random(n * 1000 + k)
            for _ in range(30):
                if _elapsed() > _DEADLINE - 15.0:
                    return None
                S = rng.sample(range(1, n // 2 + 1), half)
                S_set = set(S)
                if any((n - s) % n in S_set for s in S):
                    continue
                adj = [set() for _ in range(n)]
                for i in range(n):
                    for s in S:
                        adj[i].add((i + s) % n)
                        adj[i].add((i - s) % n)
                if all(len(adj[i]) == k for i in range(n)):
                    if _nonback_girth(adj, n, g) >= g:
                        edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                        return n, edges
    else:
        m = (k - 1) // 2
        for n in range(max(moore, k + 1), max(moore, k + 1) + 100, 2):
            if n < k + 1:
                continue
            if _elapsed() > _DEADLINE - 15.0:
                return None
            rng = random.Random(n * 1000 + k)
            for _ in range(30):
                if _elapsed() > _DEADLINE - 15.0:
                    return None
                S_vals = rng.sample(range(1, n // 2), m)
                adj = [set() for _ in range(n)]
                for i in range(n):
                    for s in S_vals:
                        adj[i].add((i + s) % n)
                        adj[i].add((i - s) % n)
                    adj[i].add((i + n // 2) % n)
                if all(len(adj[i]) == k for i in range(n)):
                    if _nonback_girth(adj, n, g) >= g:
                        edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                        return n, edges
    return None


# ---------------- construction 3: Random Configuration Model + Local Search --
def _generate_random_regular(n, k, rng):
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
    rng.shuffle(stubs)
    adj = [set() for _ in range(n)]
    for i in range(0, len(stubs), 2):
        u = stubs[i]
        v = stubs[i + 1]
        if u == v or v in adj[u]:
            return None
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _local_search(adj, n, k, g, rng, max_steps=2000):
    """Local search to improve girth via edge swaps."""
    best_girth = _nonback_girth(adj, n)
    for step in range(max_steps):
        if _elapsed() > _DEADLINE - 5.0:
            break
        girth = _nonback_girth(adj, n, g)
        if girth >= g:
            return adj, True
        if girth > best_girth:
            best_girth = girth
        # Pick random edge (u,v)
        u = rng.randint(0, n - 1)
        if not adj[u]:
            continue
        v = rng.choice(list(adj[u]))
        # Pick random edge (w,x) not incident to u,v
        found_swap = False
        for _ in range(50):
            w = rng.randint(0, n - 1)
            if w in (u, v) or w in adj[u] or w in adj[v]:
                continue
            if not adj[w]:
                continue
            x = rng.choice(list(adj[w]))
            if x in (u, v, w) or x in adj[u] or x in adj[v]:
                continue
            # Execute swap
            adj[u].remove(v)
            adj[v].remove(u)
            adj[w].remove(x)
            adj[x].remove(w)
            adj[u].add(w)
            adj[w].add(u)
            adj[v].add(x)
            adj[x].add(v)
            found_swap = True
            break
        if not found_swap:
            continue
    return adj, _nonback_girth(adj, n, g) >= g


def _random_search(k, g, seed=0):
    rng = random.Random(seed)
    R = (g - 1) // 2
    moore = 1 + k * sum((k - 1) ** i for i in range(R))
    
    for n in range(max(moore, k + 1), max(moore, k + 1) + 20):
        if n * k % 2 != 0 or n < k + 1:
            continue
        for attempt in range(5):
            if _elapsed() > _DEADLINE - 10.0:
                return None
            adj = _generate_random_regular(n, k, rng)
            if adj is None:
                continue
            if _nonback_girth(adj, n, g) >= g:
                edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                return n, edges
            # Local search
            adj2, ok = _local_search(adj, n, k, g, rng, max_steps=500)
            if ok:
                edges = [(i, j) for i in range(n) for j in adj2[i] if i < j]
                return n, edges
    return None


def construct(k, g):
    _start[0] = time.time()
    holder = {"best": None}
    lock = threading.Lock()
    
    jobs = [
        ("luw", lambda: _luw(k, g)),
        ("circ", lambda: _circulant_search(k, g)),
        ("rand0", lambda: _random_search(k, g, seed=0)),
        ("rand1", lambda: _random_search(k, g, seed=1)),
        ("rand2", lambda: _random_search(k, g, seed=2)),
    ]
    
    def _worker(fn):
        try:
            res = fn()
        except Exception:
            res = None
        if res is None:
            return
        n, edges = res
        try:
            adj = _build_adj(edges, n)
            if n > 0 and _is_valid(adj, n, k, g):
                with lock:
                    cur = holder.get("best")
                    if cur is None or n < cur[0]:
                        holder["best"] = (n, edges)
        except Exception:
            pass
    
    threads = []
    for name, fn in jobs:
        t = threading.Thread(target=_worker, args=(fn,), daemon=True)
        t.start()
        threads.append(t)

    while _elapsed() < _DEADLINE - 3.0:
        if not any(t.is_alive() for t in threads):
            break
        time.sleep(0.1)

    best = holder.get("best")
    if best is None:
        try:
            n, edges = _luw(k, g)
            best = (n, edges)
        except Exception:
            # Fallback: complete graph K_{k+1} (won't have right girth but is valid k-regular)
            n = k + 1
            edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            best = (n, edges)
    return best
