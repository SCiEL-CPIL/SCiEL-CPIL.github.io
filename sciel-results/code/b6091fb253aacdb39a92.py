import time
import numpy as np
from array import array

def _xorshift64(state):
    state ^= (state << 13) & 0xFFFFFFFFFFFFFFFF
    state ^= state >> 7
    state ^= (state << 17) & 0xFFFFFFFFFFFFFFFF
    return state & 0xFFFFFFFFFFFFFFFF

def _randrange(rng, n):
    return _xorshift64(rng) % n

def _pairing_regular(n, k, rng):
    m = n * k
    stubs = array('I', (i for i in range(n) for _ in range(k)))
    perm = list(range(m))
    for i in range(m - 1, 0, -1):
        j = _randrange(rng, i + 1)
        perm[i], perm[j] = perm[j], perm[i]
    adj = [[] for _ in range(n)]
    seen = [set() for _ in range(n)]
    for a in range(0, m, 2):
        u = stubs[perm[a]]
        v = stubs[perm[a + 1]]
        if u == v:
            return None
        if v in seen[u]:
            return None
        seen[u].add(v)
        seen[v].add(u)
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _swap_regular(adj, n, k, rng):
    u = _randrange(rng, n)
    v = adj[u][_randrange(rng, k)]
    x = _randrange(rng, n)
    y = adj[x][_randrange(rng, k)]
    if (u == x and v == y) or (u == y and v == x):
        return False
    if u == x or v == y:
        return False
    if x in adj[u] or y in adj[v]:
        return False
    adj[u].remove(v)
    adj[v].remove(u)
    adj[x].remove(y)
    adj[y].remove(x)
    adj[u].append(x)
    adj[x].append(u)
    adj[v].append(y)
    adj[y].append(v)
    return True

def _girth_check(adj, g):
    """Check if graph has girth >= g using non-backtracking walks (vectorized)."""
    n = len(adj)
    if n == 0:
        return True
    # Convert to numpy for speed
    adj_arr = np.array([np.array(adj[i], dtype=np.int32) for i in range(n)])
    
    for start in range(n):
        # dist[prev, curr] = minimum distance
        dist = np.full((n, n), -1, dtype=np.int32)
        
        # Initialize
        active = adj_arr[start].tolist()
        for nb in active:
            dist[start, nb] = 1
        
        # BFS up to depth g-1
        for depth in range(1, g - 1):
            if not active:
                break
            next_active = []
            for idx in active:
                curr = idx
                prevs = adj_arr[curr]
                for nxt in prevs:
                    if nxt == start and depth + 1 < g:
                        return False
                    if dist[curr, nxt] == -1:
                        dist[curr, nxt] = depth + 1
                        next_active.append(nxt)
            active = next_active
    
    return True

def _girth_check_fast(adj, g):
    """Faster girth check using array-based non-backtracking BFS."""
    n = len(adj)
    if n == 0:
        return True
    
    for start in range(n):
        # Use flat array for dist: dist[prev * n + curr]
        INF = 1 << 28
        dist = [INF] * (n * n)
        
        # Initialize from start
        active = []
        for nb in adj[start]:
            idx = start * n + nb
            if dist[idx] == INF:
                dist[idx] = 1
                active.append(idx)
        
        # BFS up to depth g-1
        for depth in range(1, g - 1):
            if not active:
                break
            next_active = []
            for idx in active:
                curr = idx % n
                for nxt in adj[curr]:
                    if nxt == start and depth + 1 < g:
                        return False
                    nidx = curr * n + nxt
                    if dist[nidx] == INF:
                        dist[nidx] = depth + 1
                        next_active.append(nidx)
            active = next_active
    
    return True

def _luw_fallback(k, g):
    """LUW construction as fallback."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
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

def construct(k, g):
    deadline = time.time() + 115.0
    rng = 0x9E3779B97F4A7C15
    best = None
    best_n = None

    def consider(n, adj):
        nonlocal best, best_n
        if _girth_check_fast(adj, g):
            best = adj
            best_n = n
            return True
        return False

    # Moore bound for k-regular graph with girth g
    if g % 2 == 1:
        half = (g - 1) // 2
        moore = 1 + k * sum((k - 1) ** i for i in range(half))
    else:
        half = g // 2
        moore = 2 * sum((k - 1) ** i for i in range(half))
    
    # Start from Moore bound
    n_start = max(2 * k, moore)
    if n_start % 2 != 0:
        n_start += 1
    
    # Search strategy: try multiple n values with random restarts
    # For k=5, g=9: moore = 106, search up to ~500
    
    max_n = 500
    n_values = list(range(n_start, max_n + 1, 2))
    
    # Shuffle to try diverse sizes
    import random
    random.seed(42)
    random.shuffle(n_values)
    # But try smaller ones first (more likely to succeed)
    n_values.sort()
    
    for n in n_values:
        if time.time() > deadline - 10:
            break
        
        # Try multiple random graphs at this size
        max_attempts = 30
        found_valid = False
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            adj = _pairing_regular(n, k, rng)
            if adj is None:
                continue
            
            if consider(n, adj):
                return n, [(i, j) for i in range(n) for j in adj[i] if i < j]
            
            # Try 2-switches to improve girth
            swaps = 0
            while time.time() < deadline - 3 and swaps < 200:
                if _swap_regular(adj, n, k, rng):
                    swaps += 1
                    if consider(n, adj):
                        return n, [(i, j) for i in range(n) for j in adj[i] if i < j]
        
        # Move to next n
        continue
    
    if best is not None:
        return best_n, [(i, j) for i in range(best_n) for j in best[i] if i < j]

    # Fallback: LUW
    return _luw_fallback(k, g)
