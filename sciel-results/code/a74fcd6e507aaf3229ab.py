import time
import threading
import random
from collections import deque

_DEADLINE = 118.0
_start = [None]


def _elapsed():
    s = _start[0]
    return 0.0 if s is None else time.time() - s


def _nonback_girth(adj, n):
    """Return length of shortest cycle, certified by non-backtracking walks
    from every vertex. >= g means no cycle shorter than g."""
    # We use a BFS from each directed edge (u, v) to find the shortest cycle back to u.
    # But a simpler way: BFS from each vertex u, tracking (node, parent).
    # If we reach a node (v, p) where p != u and we have seen (v, p') with p' != p, 
    # it's not necessarily a cycle unless the paths meet.
    # The standard non-backtracking BFS is:
    # For each u, for each v in adj[u]:
    #   BFS from v with parent u.
    #   If we reach u, cycle length is dist + 1.
    #   If we reach (w, v) where (w, v) was seen before from a different path, 
    #   it implies a cycle.
    
    # Optimization: We only need to check up to depth g-1.
    # If no cycle found up to depth g-1, girth >= g.
    
    for u in range(n):
        # BFS from each neighbor of u
        for v in adj[u]:
            # Queue: (node, parent, dist)
            q = deque()
            # Start: (v, u, 1)
            q.append((v, u, 1))
            visited = set()
            visited.add((v, u))
            
            while q:
                node, parent, dist = q.popleft()
                if dist >= 8: # Hard limit for speed, g is usually small
                    continue
                for w in adj[node]:
                    if w == parent:
                        continue
                    if w == u:
                        return dist + 1
                    if (w, node) in visited:
                        continue
                    visited.add((w, node))
                    q.append((w, node, dist + 1))
    return 10**9


def _has_short_cycle(adj, n, g):
    """Fast check: does the graph have any cycle of length < g?"""
    # BFS from each vertex up to depth g//2
    # If we encounter a vertex at depth d1 and d2 with d1+d2 < g, and the paths are disjoint except at start, 
    # it's a cycle.
    # Simpler: Non-backtracking BFS from each directed edge.
    for u in range(n):
        for v in adj[u]:
            q = deque()
            q.append((v, u, 1))
            visited = set()
            visited.add((v, u))
            while q:
                node, parent, dist = q.popleft()
                if dist >= g - 1:
                    continue
                if node == u:
                    return True
                for w in adj[node]:
                    if w == parent:
                        continue
                    if w == u:
                        return True
                    if (w, node) in visited:
                        continue
                    visited.add((w, node))
                    q.append((w, node, dist + 1))
    return False


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


def _build_adj(edges, n):
    adj = [set() for _ in range(n)]
    for (i, j) in edges:
        adj[i].add(j)
        adj[j].add(i)
    for i in range(n):
        adj[i] = sorted(adj[i])
    return adj


def _moore_bound(k, g):
    R = (g - 1) // 2
    return 1 + k * sum((k - 1) ** i for i in range(R))


def _luw(k, g):
    # Lazebnik-Ustimenko-Woldar construction
    # This produces a k-regular graph with girth at least g.
    # It is not always minimal, but it's a valid construction.
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
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
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


def _circulant_search(k, g, n_start, n_end, seed=0):
    rng = random.Random(seed)
    for n in range(n_start, n_end):
        if n * k % 2 != 0:
            continue
        if n < k + 1:
            continue
        
        if k % 2 == 0:
            half = k // 2
            # Try random samples of offsets
            for _ in range(200):
                if _elapsed() > _DEADLINE - 5.0:
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
                    if not _has_short_cycle(adj, n, g):
                        edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                        return n, edges
        else:
            m = (k - 1) // 2
            for _ in range(200):
                if _elapsed() > _DEADLINE - 5.0:
                    return None
                S = rng.sample(range(1, n // 2), m)
                adj = [set() for _ in range(n)]
                for i in range(n):
                    for s in S:
                        adj[i].add((i + s) % n)
                        adj[i].add((i - s) % n)
                    adj[i].add((i + n // 2) % n)
                if all(len(adj[i]) == k for i in range(n)):
                    if not _has_short_cycle(adj, n, g):
                        edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                        return n, edges
    return None


def _random_search(k, g, n, seed=0):
    rng = random.Random(seed)
    for attempt in range(50):
        if _elapsed() > _DEADLINE - 5.0:
            return None
        
        # Generate random k-regular graph
        adj = None
        for _ in range(20):
            adj = _generate_random_regular(n, k, rng)
            if adj is not None:
                break
        if adj is None:
            continue
            
        if not _has_short_cycle(adj, n, g):
            edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
            return n, edges
            
        # Local search
        for step in range(1000):
            if _elapsed() > _DEADLINE - 5.0:
                return None
            if not _has_short_cycle(adj, n, g):
                edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                return n, edges
                
            # Pick a random edge (u, v)
            u = rng.randint(0, n - 1)
            v = rng.choice(list(adj[u]))
            
            # Pick another random edge (w, x)
            w = -1
            x = -1
            for _ in range(50):
                w = rng.randint(0, n - 1)
                if w in (u, v) or w in adj[u] or w in adj[v]:
                    continue
                if not adj[w]:
                    continue
                x = rng.choice(list(adj[w]))
                if x in (u, v, w) or x in adj[u] or x in adj[v]:
                    continue
                if w in adj[u] or x in adj[v]:
                    continue
                break
            if w == -1:
                continue
                
            # Swap
            adj[u].remove(v)
            adj[v].remove(u)
            adj[w].remove(x)
            adj[x].remove(w)
            
            adj[u].add(w)
            adj[w].add(u)
            adj[v].add(x)
            adj[x].add(v)
    return None


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


def construct(k, g):
    _start[0] = time.time()
    holder = {"best": None}
    
    moore = _moore_bound(k, g)
    
    # Define search jobs
    # 1. LUW (deterministic, fast)
    jobs = [
        ("luw", lambda: _luw(k, g)),
    ]
    
    # 2. Circulant and Random searches for n = moore, moore+1, ...
    # We will run a few threads for different n ranges
    for i in range(5):
        n_start = moore + i
        n_end = moore + i + 5
        jobs.append((f"circ_{i}", lambda i=i: _circulant_search(k, g, moore + i, moore + i + 5, seed=i)))
        jobs.append((f"rand_{i}", lambda i=i: _random_search(k, g, moore + i, seed=i)))
        
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
        # Fallback
        try:
            n, edges = _luw(k, g)
            best = (n, edges)
        except Exception:
            n = k + 1
            edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            best = (n, edges)
    return best


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
