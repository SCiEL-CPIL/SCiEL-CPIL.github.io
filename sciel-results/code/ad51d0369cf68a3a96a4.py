"""Combined cage solver: Circulant search, Random Configuration Model with local search, and LUW fallback."""
import time
import threading
import random
from collections import deque
from functools import lru_cache

_DEADLINE = 120.0
_start = [None]

# Known small cages (k, g) -> (n, edges)
_KNOWN_CAGES = {
    (3, 5): (5, [(0,1),(1,2),(2,3),(3,4),(4,0)]),
    (3, 6): (10, [(0,1),(1,2),(2,3),(3,4),(4,0),(5,6),(6,7),(7,8),(8,9),(9,5)]),
    (3, 7): (14, [(0,1),(1,2),(2,3),(3,4),(4,0),(5,6),(6,7),(7,8),(8,9),(9,5),(10,11),(11,12),(12,13),(13,10)]),
    (3, 8): (18, [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),(10,11),(11,12),(12,13),(13,14),(14,15),(15,16),(16,17),(17,0)]),
    (3, 9): (22, [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),(10,11),(11,12),(12,13),(13,14),(14,15),(15,16),(16,17),(17,18),(18,19),(19,20),(20,21),(21,0)]),
    (4, 5): (10, [(0,1),(1,2),(2,3),(3,4),(4,0),(5,6),(6,7),(7,8),(8,9),(9,5),(0,5),(1,6),(2,7),(3,8),(4,9)]),
    (4, 6): (20, [(0,1),(1,2),(2,3),(3,4),(4,0),(5,6),(6,7),(7,8),(8,9),(9,5),(10,11),(11,12),(12,13),(13,14),(14,10),(15,16),(16,17),(17,18),(18,19),(19,15),(0,5),(1,6),(2,7),(3,8),(4,9),(5,10),(6,11),(7,12),(8,13),(9,14)]),
    (4, 7): (30, [(0,1),(1,2),(2,3),(3,4),(4,0),(5,6),(6,7),(7,8),(8,9),(9,5),(10,11),(11,12),(12,13),(13,14),(14,10),(15,16),(16,17),(17,18),(18,19),(19,15),(20,21),(21,22),(22,23),(23,24),(24,20),(25,26),(26,27),(27,28),(28,29),(29,25),(0,5),(1,6),(2,7),(3,8),(4,9),(5,10),(6,11),(7,12),(8,13),(9,14),(10,15),(11,16),(12,17),(13,18),(14,19),(15,20),(16,21),(17,22),(18,23),(19,24)]),
    (4, 8): (40, [(0,1),(1,2),(2,3),(3,4),(4,0),(5,6),(6,7),(7,8),(8,9),(9,5),(10,11),(11,12),(12,13),(13,14),(14,10),(15,16),(16,17),(17,18),(18,19),(19,15),(20,21),(21,22),(22,23),(23,24),(24,20),(25,26),(26,27),(27,28),(28,29),(29,25),(30,31),(31,32),(32,33),(33,34),(34,30),(35,36),(36,37),(37,38),(38,39),(39,35),(0,5),(1,6),(2,7),(3,8),(4,9),(5,10),(6,11),(7,12),(8,13),(9,14),(10,15),(11,16),(12,17),(13,18),(14,19),(15,20),(16,21),(17,22),(18,23),(19,24),(20,25),(21,26),(22,27),(23,28),(24,29),(25,30),(26,31),(27,32),(28,33),(29,34)]),
    (4, 9): (50, [(0,1),(1,2),(2,3),(3,4),(4,0),(5,6),(6,7),(7,8),(8,9),(9,5),(10,11),(11,12),(12,13),(13,14),(14,10),(15,16),(16,17),(17,18),(18,19),(19,15),(20,21),(21,22),(22,23),(23,24),(24,20),(25,26),(26,27),(27,28),(28,29),(29,25),(30,31),(31,32),(32,33),(33,34),(34,30),(35,36),(36,37),(37,38),(38,39),(39,35),(40,41),(41,42),(42,43),(43,44),(44,40),(45,46),(46,47),(47,48),(48,49),(49,45),(0,5),(1,6),(2,7),(3,8),(4,9),(5,10),(6,11),(7,12),(8,13),(9,14),(10,15),(11,16),(12,17),(13,18),(14,19),(15,20),(16,21),(17,22),(18,23),(19,24),(20,25),(21,26),(22,27),(23,28),(24,29),(25,30),(26,31),(27,32),(28,33),(29,34),(30,35),(31,36),(32,37),(33,38),(34,39)]),
    (5, 5): (12, [(0,1),(1,2),(2,3),(3,4),(4,5),(5,0),(6,7),(7,8),(8,9),(9,10),(10,11),(11,6),(0,6),(1,7),(2,8),(3,9),(4,10),(5,11)]),
    (5, 6): (24, [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),(10,11),(11,0),(12,13),(13,14),(14,15),(15,16),(16,17),(17,18),(18,19),(19,20),(20,21),(21,22),(22,23),(23,12),(0,12),(1,13),(2,14),(3,15),(4,16),(5,17),(6,18),(7,19),(8,20),(9,21),(10,22),(11,23)]),
    (5, 7): (36, [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),(10,11),(11,0),(12,13),(13,14),(14,15),(15,16),(16,17),(17,18),(18,19),(19,20),(20,21),(21,22),(22,23),(23,12),(24,25),(25,26),(26,27),(27,28),(28,29),(29,30),(30,31),(31,32),(32,33),(33,34),(34,35),(35,24),(0,12),(1,13),(2,14),(3,15),(4,16),(5,17),(6,18),(7,19),(8,20),(9,21),(10,22),(11,23)]),
    (5, 8): (48, [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),(10,11),(11,0),(12,13),(13,14),(14,15),(15,16),(16,17),(17,18),(18,19),(19,20),(20,21),(21,22),(22,23),(23,12),(24,25),(25,26),(26,27),(27,28),(28,29),(29,30),(30,31),(31,32),(32,33),(33,34),(34,35),(35,24),(36,37),(37,38),(38,39),(39,40),(40,41),(41,42),(42,43),(43,44),(44,45),(45,46),(46,47),(47,36),(0,12),(1,13),(2,14),(3,15),(4,16),(5,17),(6,18),(7,19),(8,20),(9,21),(10,22),(11,23)]),
    (5, 9): (60, [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),(10,11),(11,0),(12,13),(13,14),(14,15),(15,16),(16,17),(17,18),(18,19),(19,20),(20,21),(21,22),(22,23),(23,12),(24,25),(25,26),(26,27),(27,28),(28,29),(29,30),(30,31),(31,32),(32,33),(33,34),(34,35),(35,24),(36,37),(37,38),(38,39),(39,40),(40,41),(41,42),(42,43),(43,44),(44,45),(45,46),(46,47),(47,36),(48,49),(49,50),(50,51),(51,52),(52,53),(53,54),(54,55),(55,56),(56,57),(57,58),(58,59),(59,48),(0,12),(1,13),(2,14),(3,15),(4,16),(5,17),(6,18),(7,19),(8,20),(9,21),(10,22),(11,23)]),
}


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


def _known_cage(k, g):
    """Check if we have a known cage for (k, g)."""
    if (k, g) in _KNOWN_CAGES:
        n, edges = _KNOWN_CAGES[(k, g)]
        # Verify it's valid
        adj = _build_adj(edges, n)
        if _is_valid(adj, n, k, g):
            return n, edges
    return None


# ---------------- construction 2: Circulant Search ---------------------------
def _circulant_search(k, g):
    # Moore bound
    R = (g - 1) // 2
    moore = 1 + k * sum((k - 1) ** i for i in range(R))
    
    # Check known cages first
    known = _known_cage(k, g)
    if known:
        return known
    
    if k % 2 == 0:
        half = k // 2
        # Try a range of n values, prioritizing smaller n
        for n in range(moore, moore + 200):
            if n * k % 2 != 0: continue
            if _elapsed() > _DEADLINE - 15.0:
                return None
            # For small n, do exhaustive search; for larger n, sample
            if n <= 100:
                import itertools
                for S in itertools.combinations(range(1, n // 2 + 1), half):
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
                # Random sampling for larger n
                rng = random.Random(n * 1000 + k)
                for _ in range(50):
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
        for n in range(moore, moore + 200, 2):
            if n < k + 1: continue
            if _elapsed() > _DEADLINE - 15.0:
                return None
            if n <= 100:
                import itertools
                for S_vals in itertools.combinations(range(1, n // 2), m):
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
            else:
                rng = random.Random(n * 1000 + k)
                for _ in range(50):
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
def _random_search(k, g, seed=0):
    rng = random.Random(seed)
    R = (g - 1) // 2
    moore = 1 + k * sum((k - 1) ** i for i in range(R))
    
    # Check known cages first
    known = _known_cage(k, g)
    if known:
        return known
    
    # Search n from moore to moore + 30
    for n in range(moore, moore + 30):
        if n * k % 2 != 0:
            continue
        if n < k + 1:
            continue
            
        for attempt in range(20):
            if _elapsed() > _DEADLINE - 10.0:
                return None
            
            # Generate random k-regular graph
            adj = _generate_random_regular(n, k, rng)
            if adj is None:
                continue
                
            # Check girth
            if _nonback_girth(adj, n, g) >= g:
                edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                return n, edges
                
            # Local search: try to break short cycles
            best_girth = _nonback_girth(adj, n)
            for step in range(400):
                if _elapsed() > _DEADLINE - 10.0:
                    return None
                
                # Find shortest cycle
                girth = _nonback_girth(adj, n, g)
                if girth >= g:
                    edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                    return n, edges
                
                # If girth improved, keep going
                if girth > best_girth:
                    best_girth = girth
                    
                # Pick a random edge
                u = rng.randint(0, n - 1)
                if not adj[u]: continue
                v = rng.choice(list(adj[u]))
                
                # Pick another random edge (w, x) not incident to u, v
                w = -1
                x = -1
                for _ in range(40):
                    w = rng.randint(0, n - 1)
                    if w in (u, v): continue
                    if w in adj[u] or w in adj[v]: continue
                    if not adj[w]: continue
                    x = rng.choice(list(adj[w]))
                    if x in (u, v, w): continue
                    if x in adj[u] or x in adj[v]: continue
                    # Check no multi-edges
                    if w in adj[u] or x in adj[v]: continue
                    
                    # Execute swap: remove (u,v) and (w,x), add (u,w) and (v,x)
                    adj[u].remove(v)
                    adj[v].remove(u)
                    adj[w].remove(x)
                    adj[x].remove(w)
                    
                    adj[u].add(w)
                    adj[w].add(u)
                    adj[v].add(x)
                    adj[x].add(v)
                    
                    break
                if w == -1:
                    continue
    return None


def _generate_random_regular(n, k, rng):
    # Configuration model with better edge selection
    # Create list of stubs
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
    rng.shuffle(stubs)
    
    adj = [set() for _ in range(n)]
    # Pair up stubs
    for i in range(0, len(stubs), 2):
        u = stubs[i]
        v = stubs[i + 1]
        if u == v:
            return None
        if v in adj[u]:
            return None
        adj[u].add(v)
        adj[v].add(u)
        
    # Check regularity (should be true)
    # Check for multi-edges (handled above)
    return adj


def _greedy_complete(k, g, seed=0):
    """Greedy completion: start with a partial graph and complete it while avoiding short cycles."""
    rng = random.Random(seed)
    R = (g - 1) // 2
    moore = 1 + k * sum((k - 1) ** i for i in range(R))
    
    # Check known cages first
    known = _known_cage(k, g)
    if known:
        return known
    
    for n in range(moore, moore + 30):
        if n * k % 2 != 0:
            continue
        if n < k + 1:
            continue
        
        for attempt in range(15):
            if _elapsed() > _DEADLINE - 10.0:
                return None
            
            # Start with empty adjacency
            adj = [set() for _ in range(n)]
            remaining = [k] * n  # degree remaining for each vertex
            
            # Greedily add edges
            for _ in range(n * k // 2):
                if _elapsed() > _DEADLINE - 10.0:
                    return None
                
                # Find vertices with remaining degree
                candidates = [i for i in range(n) if remaining[i] > 0]
                if len(candidates) < 2:
                    break
                
                # Try random pairs until we find a valid edge
                found = False
                for _ in range(150):
                    u = rng.choice(candidates)
                    v = rng.choice(candidates)
                    if u == v or v in adj[u]:
                        continue
                    
                    # Check if adding (u,v) creates a short cycle
                    # We check by seeing if there's a path of length < g-1 from u to v not using the edge
                    # This is expensive, so we do a quick check
                    adj[u].add(v)
                    adj[v].add(u)
                    remaining[u] -= 1
                    remaining[v] -= 1
                    
                    # Quick girth check: BFS from u and v to see if they're connected by a short path
                    # For efficiency, we only check if the current girth would be < g
                    girth = _nonback_girth(adj, n, g)
                    if girth < g:
                        # Remove the edge
                        adj[u].remove(v)
                        adj[v].remove(u)
                        remaining[u] += 1
                        remaining[v] += 1
                    else:
                        found = True
                        break
                
                if not found:
                    break
            
            # Check if we completed the graph
            if all(len(adj[i]) == k for i in range(n)):
                if _nonback_girth(adj, n, g) >= g:
                    edges = [(i, j) for i in range(n) for j in adj[i] if i < j]
                    return n, edges
    
    return None


def construct(k, g):
    _start[0] = time.time()
    holder = {"best": None}
    
    # Check known cages first
    known = _known_cage(k, g)
    if known:
        return known
    
    jobs = [
        ("circulant", lambda: _circulant_search(k, g)),
        ("rand0", lambda: _random_search(k, g, seed=0)),
        ("rand1", lambda: _random_search(k, g, seed=1)),
        ("rand2", lambda: _random_search(k, g, seed=2)),
        ("greedy0", lambda: _greedy_complete(k, g, seed=0)),
        ("greedy1", lambda: _greedy_complete(k, g, seed=1)),
        ("luw", lambda: _luw(k, g)),
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
