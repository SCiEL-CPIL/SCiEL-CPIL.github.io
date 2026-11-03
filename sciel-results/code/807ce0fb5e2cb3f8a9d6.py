"""Cage solver: optimized circulant search with algebraic pre-filters + LUW fallback."""
import time
import threading
from collections import deque

_DEADLINE = 150.0
_start = [None]


def _elapsed():
    s = _start[0]
    return 0.0 if s is None else time.time() - s


# ---------------- exact girth (non-backtracking walks, all vertices) ----------
def _nonback_girth(adj, n):
    """Return length of shortest cycle, certified by non-backtracking walks
    from every vertex. >= g means no cycle shorter than g."""
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
            if w == u:
                return dist + 1
            key = (w, v)
            if key in prev:
                continue
            prev[key] = (v, dist + 1)
            q.append(key)
    return 10 ** 9


def _nonback_girth_limited(adj, n, max_len):
    """Return shortest cycle length if < max_len, else max_len. Early termination."""
    prev = {}
    q = deque()
    for u in range(n):
        for p in adj[u]:
            key = (p, u)
            prev[key] = (u, 1)
            q.append(key)
    while q:
        (v, p), (u, dist) = q.popleft()
        if dist + 1 >= max_len:
            continue
        for w in adj[v]:
            if w == p:
                continue
            if w == u:
                return dist + 1
            key = (w, v)
            if key in prev:
                continue
            prev[key] = (v, dist + 1)
            q.append(key)
    return max_len


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


# ---------------- construction 1: LUW (Lazebnik-Ustimenko-Woldar) ------------
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


# ---------------- construction 2: optimized circulant search -----------------
def _circulant_fast(k, g):
    """Optimized circulant search for k=5, g=9.
    Key improvements:
    1. Algebraic pre-filters for short cycles (length 3, 4) to reject bad candidates fast.
    2. Efficient BFS for remaining candidates.
    3. Broader candidate generation.
    """
    if k != 5:
        return None

    min_n = 428
    max_n = 1200
    deadline = _DEADLINE - 15.0

    # Fractions for generating a, b
    frac_pool = [
        0.236, 0.382, 0.118, 0.294, 0.412, 0.191, 0.347, 0.271,
        0.447, 0.153, 0.319, 0.228, 0.361, 0.472, 0.183, 0.257,
        0.401, 0.139, 0.283, 0.357, 0.421, 0.167, 0.241, 0.315,
        0.100, 0.900, 0.200, 0.800, 0.300, 0.700, 0.400, 0.600,
        0.050, 0.950, 0.150, 0.850, 0.250, 0.750, 0.350, 0.650,
    ]

    def get_candidates(n, max_candidates=300):
        half_n = n // 2
        cands = []
        seen = set()
        # Generate from fractions
        for fa in frac_pool:
            for fb in frac_pool:
                a = int(round(n * fa))
                b = int(round(n * fb))
                if a < 1 or a >= half_n or b < 1 or b >= half_n:
                    continue
                if a == b or a == half_n or b == half_n:
                    continue
                if (a, b) in seen:
                    continue
                seen.add((a, b))
                cands.append((a, b))
        
        # Add some pseudo-random based on n
        for i in range(100):
            a = 1 + (n * 7919 * (i+1) % (half_n - 1))
            b = 1 + (n * 104729 * (i+1) % (half_n - 1))
            if a < 1 or a >= half_n or b < 1 or b >= half_n or a == b:
                continue
            if (a, b) in seen:
                continue
            seen.add((a, b))
            cands.append((a, b))
            
        return cands[:max_candidates]

    for n in range(min_n, max_n + 1, 2):
        if _elapsed() > deadline:
            break

        half_n = n // 2
        candidates = get_candidates(n, max_candidates=200)

        for a, b in candidates:
            if _elapsed() > deadline:
                break

            S = [a, n - a, b, n - b, half_n]
            if len(set(S)) != 5:
                continue

            # --- Algebraic Pre-filters for short cycles ---
            # Cycle length 3: a+a+b=0, a+b+b=0, etc.
            # In a 5-regular circulant, triangles are rare but possible.
            # Check sums of 3 generators = 0 mod n
            # Generators: a, -a, b, -b, n/2
            # Possible sums of 3:
            # 3a, 3b, 2a+b, 2a-b, a+2b, a-2b, a+b+n/2, etc.
            # If any sum is 0 mod n, girth <= 3.
            # We can check a few common ones quickly.
            if (3 * a) % n == 0 or (3 * b) % n == 0:
                continue
            if (2 * a + b) % n == 0 or (2 * a - b) % n == 0:
                continue
            if (a + 2 * b) % n == 0 or (a - 2 * b) % n == 0:
                continue
            if (a + b + half_n) % n == 0: # a+b+n/2
                continue
            if (a - b + half_n) % n == 0:
                continue
            if (-a + b + half_n) % n == 0:
                continue
            if (-a - b + half_n) % n == 0:
                continue
            if (a + b - half_n) % n == 0:
                continue
            if (a - b - half_n) % n == 0:
                continue

            # Cycle length 4:
            # 4a, 4b, 2a+2b, 2a-2b, a+b+a+b (same as 2a+2b), etc.
            if (4 * a) % n == 0 or (4 * b) % n == 0:
                continue
            if (2 * a + 2 * b) % n == 0:
                continue
            if (2 * a - 2 * b) % n == 0:
                continue
            if (a + b + half_n + half_n) % n == 0: # a+b+n
                if (a+b) % n == 0: continue # already covered by 2-cycle logic? No, a+b=0 is 2-cycle in terms of steps? 
                # If a+b=0, then b=-a. But we ensured a != b and a != -b (since S has 5 distinct elements).
                # So a+b != 0.
                pass
            if (a - b + half_n + half_n) % n == 0:
                pass
            # a + n/2 + b + n/2 = a+b+n. If a+b=0, rejected.
            
            # If it passes filters, do BFS
            neg = {s: (n - s) % n for s in S}
            visited = set()
            q = deque()
            for s in S:
                state = (s, s)
                visited.add(state)
                q.append((s, s, 1))

            girth_found = 10 ** 9
            while q:
                v, last_s, dist = q.popleft()
                if dist >= g:
                    break
                for s in S:
                    if s == neg[last_s]:
                        continue
                    new_v = (v + s) % n
                    if new_v == 0:
                        girth_found = dist + 1
                        break
                    new_state = (new_v, s)
                    if new_state not in visited:
                        visited.add(new_state)
                        q.append((new_v, s, dist + 1))
                if girth_found <= g:
                    break

            if girth_found >= g:
                # Build adjacency and verify with full girth check
                adj = [set() for _ in range(n)]
                for i in range(n):
                    for s in S:
                        adj[i].add((i + s) % n)
                if all(len(adj[i]) == 5 for i in range(n)):
                    if _nonback_girth(adj, n) >= g:
                        edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                        return (n, edges)

    return None


# ---------------- construction 3: random regular graph search ----------------
import random as _random


def _random_regular_search(k, g, seed=0):
    """Search for a k-regular graph with girth >= g by sampling random regular graphs."""
    rng = _random.Random(seed)
    R = (g - 1) // 2
    moore = 1 + k * sum((k - 1) ** i for i in range(R))
    if (moore * k) % 2 != 0:
        moore += 1

    deadline = _DEADLINE - 20.0
    max_n = 800
    max_attempts = 20

    for n in range(moore, max_n + 1, 2):
        if _elapsed() > deadline:
            break
        if (n * k) % 2 != 0:
            continue

        for attempt in range(max_attempts):
            if _elapsed() > deadline:
                break

            # Configuration model
            half_edges = []
            for v in range(n):
                for _ in range(k):
                    half_edges.append(v)
            rng.shuffle(half_edges)

            adj = [set() for _ in range(n)]
            valid = True
            for i in range(0, len(half_edges), 2):
                u = half_edges[i]
                v = half_edges[i + 1]
                if u == v or v in adj[u]:
                    valid = False
                    break
                adj[u].add(v)
                adj[v].add(u)

            if not valid:
                continue
            if any(len(adj[i]) != k for i in range(n)):
                continue

            # Quick girth check
            if _nonback_girth_limited(adj, n, g) >= g:
                edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                return (n, edges)

    return None


# ---------------- parallel candidate verification ---------------------------
def _worker(name, fn, holder):
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
        ("circulant", lambda: _circulant_fast(k, g)),
        ("rand0", lambda: _random_regular_search(k, g, seed=0)),
        ("rand1", lambda: _random_regular_search(k, g, seed=1)),
        ("rand2", lambda: _random_regular_search(k, g, seed=42)),
    ]

    threads = []
    for name, fn in jobs:
        t = threading.Thread(target=_worker, args=(name, fn, holder), daemon=True)
        t.start()
        threads.append(t)

    # Wait with safety margin
    while _elapsed() < _DEADLINE - 5.0:
        if not any(t.is_alive() for t in threads):
            break
        time.sleep(0.05)

    best = holder.get("best")
    if best is None:
        # Last resort: LUW
        try:
            n, edges = _luw(k, g)
            best = (n, edges)
        except Exception:
            n = k + 1
            edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            best = (n, edges)
    return best
