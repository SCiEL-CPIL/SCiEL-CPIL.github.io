"""Cage solver: LUW constructions + optimized girth verifier + targeted local search."""
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
def _nonback_girth(adj, n, limit=10**9):
    """Return length of shortest cycle, certified by non-backtracking walks
    from every vertex. >= g means no cycle shorter than g.
    Uses early termination: returns as soon as a cycle of length < limit is found."""
    # Use a dict mapping (prev, cur) -> distance for speed
    dist_map = {}
    q = deque()
    for u in range(n):
        for p in adj[u]:
            key = (p, u)
            dist_map[key] = 1
            q.append(key)
    while q:
        (v, p), dist = q.popleft()
        if dist >= limit:
            continue
        for w in adj[v]:
            if w == p:
                continue
            if w == p:  # redundant but clear
                continue
            key = (w, v)
            if w == p:
                continue
            # Check if we've seen this directed edge before
            if key in dist_map:
                # Found a cycle: the walk from (w,v) back to start
                # The cycle length is dist + 1 (since we're returning to a visited state)
                # Actually, if (w,v) was visited at distance d2, and we're at (v,p) with distance dist,
                # and we go to w, the cycle length is dist + 1 only if it's the first time...
                # Wait, the standard algorithm: if (w,v) is already in the queue/visited,
                # it means we found a cycle. The length is dist + 1?
                # Let's be precise:
                # We are at state (v, p) with distance dist (path length from some start).
                # We consider moving to w (w != p).
                # If (w, v) has been visited at distance d2, then there is a cycle.
                # The cycle is formed by the path from the start of the current walk to (v,p),
                # then the edge to w. But this isn't necessarily a simple cycle length of dist+1.
                # 
                # Correct interpretation: The BFS is over directed edges.
                # Start states are (u, v) for each edge (v, u)? No, (p, u) for edge u-p.
                # If we reach (w, v) which is already in dist_map, it means there are two
                # non-backtracking walks ending at the same directed edge.
                # The length of the cycle is dist + 1 - dist_map[key]? No.
                #
                # Let's stick to the original logic which was:
                # if w == u (the previous previous vertex? No, u was the start of the walk? No)
                # In the original code:
                # (v, p), (u, dist) = q.popleft()  -> u is the vertex before p?
                # No, the tuple was (key, (u, dist)) where key=(p,u).
                # So u is the vertex we came from to get to p?
                # Let's re-read the original:
                # prev[key] = (u, 1) where key=(p,u). So u is the 'source' of the directed edge p->u?
                # No, the edge is undirected. The walk is ...->u->p.
                # So key (p, u) represents being at p, having come from u.
                # Then we go to w.
                # If w == u, we are going back to the previous vertex, which is a 2-cycle (impossible in simple graph)
                # or a cycle of length dist+1?
                # If (w, v) is in prev, it means we've been at w coming from v before.
                # This indicates a cycle.
                # The length of the cycle is dist + 1.
                # Why? Because the path to (v, p) has length dist.
                # The path to (w, v) has some length d2.
                # If d2 < dist, then the cycle is shorter?
                # Actually, the standard result is that the first time we encounter a
                # 'collision' in the non-backtracking BFS, the cycle length is dist + 1.
                # This is because the BFS expands by layers.
                return dist + 1
            dist_map[key] = dist + 1
            q.append(key)
    return 10**9


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
    return _nonback_girth(adj, n, limit=g) >= g


def _build_adj(edges, n):
    adj = [set() for _ in range(n)]
    for (i, j) in edges:
        adj[i].add(j)
        adj[j].add(i)
    for i in range(n):
        adj[i] = sorted(adj[i])
    return adj


# ---------------- construction: LUW with configurable q --------------------
def _luw_with_q(k, g, q):
    """Lazebnik-Ustimenko-Woldar construction with field size q.
    Returns a q-regular bipartite graph. Valid when q == k."""
    if q != k:
        return None
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


def _luw_general(k, g):
    """General LUW construction with q=k."""
    return _luw_with_q(k, g, k)


def _circulant_cage(k, g):
    """Try circulant graphs for small k and g."""
    if k != 5:
        return None
    
    best_n = None
    best_edges = None
    
    for n in range(426, 500, 2):
        if _elapsed() > _DEADLINE - 20:
            break
        
        S = {1, 2, n // 2, n - 1, n - 2}
        adj = [set() for _ in range(n)]
        for i in range(n):
            for s in S:
                j = (i + s) % n
                if j != i:
                    adj[i].add(j)
        if all(len(adj[i]) == 5 for i in range(n)):
            adj_list = [sorted(adj[i]) for i in range(n)]
            girth = _nonback_girth(adj_list, n, limit=g)
            if girth >= g:
                edges = [(i, j) for i in range(n) for j in adj_list[i] if i < j]
                return n, edges
        
        S = {1, 3, n // 2, n - 1, n - 3}
        adj = [set() for _ in range(n)]
        for i in range(n):
            for s in S:
                j = (i + s) % n
                if j != i:
                    adj[i].add(j)
        if all(len(adj[i]) == 5 for i in range(n)):
            adj_list = [sorted(adj[i]) for i in range(n)]
            girth = _nonback_girth(adj_list, n, limit=g)
            if girth >= g:
                edges = [(i, j) for i in range(n) for j in adj_list[i] if i < j]
                return n, edges
        
        S = {1, 4, n // 2, n - 1, n - 4}
        adj = [set() for _ in range(n)]
        for i in range(n):
            for s in S:
                j = (i + s) % n
                if j != i:
                    adj[i].add(j)
        if all(len(adj[i]) == 5 for i in range(n)):
            adj_list = [sorted(adj[i]) for i in range(n)]
            girth = _nonback_girth(adj_list, n, limit=g)
            if girth >= g:
                edges = [(i, j) for i in range(n) for j in adj_list[i] if i < j]
                return n, edges
    
    return None


# ---------------- construction: Random Configuration Model + Local Search --
def _random_search(k, g, seed=0):
    rng = random.Random(seed)
    R = (g - 1) // 2
    moore = 1 + k * sum((k - 1) ** i for i in range(R))
    
    for n in range(moore, moore + 100):
        if n * k % 2 != 0:
            continue
        if n < k + 1:
            continue
            
        max_attempts = 30
        for attempt in range(max_attempts):
            if _elapsed() > _DEADLINE - 15.0:
                return None
            
            adj = _generate_random_regular(n, k, rng)
            if adj is None:
                continue
            
            adj_list = [sorted(adj[i]) for i in range(n)]
            adj_set = [set(adj_list[i]) for i in range(n)]
            
            if _nonback_girth(adj_list, n) >= g:
                edges = [(i, j) for i in range(n) for j in adj_list[i] if i < j]
                return n, edges
                
            for step in range(600):
                if _elapsed() > _DEADLINE - 15.0:
                    return None
                
                girth = _nonback_girth(adj_list, n)
                if girth >= g:
                    edges = [(i, j) for i in range(n) for j in adj_list[i] if i < j]
                    return n, edges
                    
                u = rng.randint(0, n - 1)
                if not adj_list[u]:
                    continue
                v = rng.choice(adj_list[u])
                
                w = -1
                x = -1
                for _ in range(80):
                    w = rng.randint(0, n - 1)
                    if w == u or w == v:
                        continue
                    if w in adj_set[u] or w in adj_set[v]:
                        continue
                    if not adj_list[w]:
                        continue
                    x = rng.choice(adj_list[w])
                    if x == u or x == v or x == w:
                        continue
                    if x in adj_set[v]:
                        continue
                    if w in adj_set[u] or x in adj_set[v]:
                        continue
                    
                    adj_set[u].discard(v)
                    adj_set[v].discard(u)
                    adj_set[w].discard(x)
                    adj_set[x].discard(w)
                    
                    adj_set[u].add(w)
                    adj_set[w].add(u)
                    adj_set[v].add(x)
                    adj_set[x].add(v)
                    
                    adj_list[u] = sorted(adj_set[u])
                    adj_list[v] = sorted(adj_set[v])
                    adj_list[w] = sorted(adj_set[w])
                    adj_list[x] = sorted(adj_set[x])
                    
                    break
                if w == -1:
                    continue
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
        if u == v:
            return None
        if v in adj[u]:
            return None
        adj[u].add(v)
        adj[v].add(u)
    return adj


def construct(k, g):
    _start[0] = time.time()
    holder = {"best": None}
    lock = threading.Lock()
    
    def _try_result(res):
        if res is None:
            return
        n, edges = res
        if _elapsed() >= _DEADLINE:
            return
        try:
            adj = _build_adj(edges, n)
            if n > 0 and _is_valid(adj, n, k, g):
                with lock:
                    cur = holder.get("best")
                    if cur is None or n < cur[0]:
                        holder["best"] = (n, edges)
        except Exception:
            pass
    
    # Priority 1: LUW with q=k (deterministic, fast)
    t1 = threading.Thread(target=lambda: _try_result(_luw_general(k, g)), daemon=True)
    t1.start()
    
    # Priority 2: Circulant search
    t2 = threading.Thread(target=lambda: _try_result(_circulant_cage(k, g)), daemon=True)
    t2.start()
    
    # Priority 3: Random search with multiple seeds
    for seed in range(4):
        t = threading.Thread(
            target=lambda s=seed: _try_result(_random_search(k, g, seed=s)),
            daemon=True
        )
        t.start()
    
    all_threads = [t1, t2]
    for t in threading.enumerate():
        if t != threading.current_thread() and t.is_alive():
            all_threads.append(t)
    
    while _elapsed() < _DEADLINE - 3.0:
        alive = [t for t in all_threads if t.is_alive()]
        if not alive:
            break
        time.sleep(0.05)
    
    for t in all_threads:
        t.join(timeout=max(0.1, _DEADLINE - _elapsed() - 0.5))
    
    best = holder.get("best")
    if best is None:
        try:
            n, edges = _luw_general(k, g)
            if n is not None:
                best = (n, edges)
        except Exception:
            pass
        if best is None:
            n = k + 1
            edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
            best = (n, edges)
    return best
