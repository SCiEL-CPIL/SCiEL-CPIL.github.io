"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import random as _random
from collections import deque

def _verify_girth_fast(adj, n, g):
    """Check if graph has girth >= g using non-backtracking BFS from each vertex.
    Uses list-based adjacency for speed. Returns True if girth >= g, False otherwise."""
    for start in range(n):
        dist = {}
        dq = deque()
        dq.append((start, -1))
        dist[(start, -1)] = 0
        while dq:
            v, prev = dq.popleft()
            d = dist[(v, prev)]
            if d >= g - 1:
                continue
            for w in adj[v]:
                if w == prev:
                    continue
                if w == start and d + 1 >= 3:
                    return False
                state = (w, v)
                if state not in dist:
                    dist[state] = d + 1
                    dq.append((w, v))
    return True

def _build_lift_petersen(m, seed=0):
    """Build a voltage graph lift of the Petersen graph over Z_m.
    Returns (n, adj, edges) or None if invalid."""
    rng = _random.Random(seed)
    
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    
    n = 10 * m
    adj = [set() for _ in range(n)]
    edges = []
    
    voltages = {}
    for u, v in petersen_edges:
        voltages[(u, v)] = rng.randint(0, m - 1)
        voltages[(v, u)] = (-voltages[(u, v)]) % m
    
    for u, v in petersen_edges:
        vol = voltages[(u, v)]
        for i in range(m):
            a = u * m + i
            b = v * m + (i + vol) % m
            if b not in adj[a]:
                adj[a].add(b)
                adj[b].add(a)
                edges.append((a, b))
    
    for i in range(n):
        if len(adj[i]) != 3:
            return None
    
    return n, [list(s) for s in adj], edges

def _build_lift_k33(m, seed=0):
    """Build a voltage graph lift of K_{3,3} over Z_m.
    K_{3,3}: 6 vertices, 9 edges, 3-regular, bipartite.
    Returns (n, adj, edges) or None if invalid."""
    rng = _random.Random(seed)
    
    k33_edges = [
        (0, 3), (0, 4), (0, 5),
        (1, 3), (1, 4), (1, 5),
        (2, 3), (2, 4), (2, 5)
    ]
    
    n = 6 * m
    adj = [set() for _ in range(n)]
    edges = []
    
    voltages = {}
    for u, v in k33_edges:
        voltages[(u, v)] = rng.randint(0, m - 1)
        voltages[(v, u)] = (-voltages[(u, v)]) % m
    
    for u, v in k33_edges:
        vol = voltages[(u, v)]
        for i in range(m):
            a = u * m + i
            b = v * m + (i + vol) % m
            if b not in adj[a]:
                adj[a].add(b)
                adj[b].add(a)
                edges.append((a, b))
    
    for i in range(n):
        if len(adj[i]) != 3:
            return None
    
    return n, [list(s) for s in adj], edges

def _build_lift_prism(m, seed=0):
    """Build a voltage graph lift of the triangular prism (6 vertices, 9 edges, 3-regular) over Z_m.
    Returns (n, adj, edges) or None if invalid."""
    rng = _random.Random(seed)
    
    # Triangular prism: two triangles connected by matching
    prism_edges = [
        (0, 1), (1, 2), (2, 0),  # top triangle
        (3, 4), (4, 5), (5, 3),  # bottom triangle
        (0, 3), (1, 4), (2, 5)   # vertical edges
    ]
    
    n = 6 * m
    adj = [set() for _ in range(n)]
    edges = []
    
    voltages = {}
    for u, v in prism_edges:
        voltages[(u, v)] = rng.randint(0, m - 1)
        voltages[(v, u)] = (-voltages[(u, v)]) % m
    
    for u, v in prism_edges:
        vol = voltages[(u, v)]
        for i in range(m):
            a = u * m + i
            b = v * m + (i + vol) % m
            if b not in adj[a]:
                adj[a].add(b)
                adj[b].add(a)
                edges.append((a, b))
    
    for i in range(n):
        if len(adj[i]) != 3:
            return None
    
    return n, [list(s) for s in adj], edges

def _try_lift(base_builder, m, g, max_attempts=100):
    """Try multiple random voltage assignments for a given lift size m.
    Returns (n, edges) if a valid graph is found, else None."""
    for attempt in range(max_attempts):
        result = base_builder(m, seed=attempt)
        if result is None:
            continue
        n, adj, edges = result
        if _verify_girth_fast(adj, n, g):
            return n, edges
    return None

def construct(k, g):
    start_time = _time.time()
    deadline = start_time + 115
    
    if k != 3:
        # Fallback to LUW for other k values
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
            add, sub, mul = (lambda x,y: x^y), (lambda x,y: x^y), (lambda x,y: table[x][y])
        else:
            add, sub, mul = (lambda x,y:(x+y)%q), (lambda x,y:(x-y)%q), (lambda x,y:x*y%q)
        rule = {2:(1,1), 3:(2,1), 4:(1,2)}
        for c in range(5, d+1):
            r, c0 = (c-5)%4, c-(c-5)%4
            rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0]*(d-1)
                for c in range(2, d+1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side==0 else sub(x[c], mul(x[a], y[b]))
                yield 1-side, tuple(y)
        start = (0, (None,) + (0,)*d)
        index, order, edges = {start:0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges
    
    # For k=3: Use voltage graph lifts
    if g % 2 == 0:
        moore = 2 * (2**(g//2) - 1)
    else:
        moore = 2 * (2**(g//2 - 1) - 1) + 2
    
    best_result = None
    
    # Try different base graphs and lift sizes
    # K_{3,3} lift: n = 6m, Petersen lift: n = 10m, Prism lift: n = 6m
    # We want to find the smallest n >= moore that works
    
    # Strategy: try m values starting from minimum needed for each base graph
    # For K_{3,3}: m_min = ceil(moore / 6)
    # For Petersen: m_min = ceil(moore / 10)
    # For Prism: m_min = ceil(moore / 6)
    
    builders = [
        (_build_lift_k33, 6),
        (_build_lift_prism, 6),
        (_build_lift_petersen, 10),
    ]
    
    # Try each builder with increasing m
    for builder, base_size in builders:
        if _time.time() > deadline - 10:
            break
        m_min = max(1, (moore + base_size - 1) // base_size)
        # Try a range of m values
        for m in range(m_min, m_min + 50):
            if _time.time() > deadline - 5:
                break
            n = base_size * m
            if n < moore:
                continue
            result = _try_lift(builder, m, g, max_attempts=50)
            if result is not None:
                best_result = result
                return n, edges
    
    # If no lift worked, try random 3-regular graphs
    if _time.time() < deadline - 10:
        n_try = moore
        if n_try % 2 != 0:
            n_try += 1
        while n_try < 2000 and _time.time() < deadline - 10:
            for attempt in range(20):
                if _time.time() > deadline - 5:
                    break
                stubs = []
                for v in range(n_try):
                    for _ in range(3):
                        stubs.append(v)
                _random.shuffle(stubs)
                adj = [set() for _ in range(n_try)]
                edges = []
                valid = True
                for i in range(0, n_try * 3, 2):
                    u, v = stubs[i], stubs[i + 1]
                    if u == v or v in adj[u]:
                        valid = False
                        break
                    adj[u].add(v)
                    adj[v].add(u)
                    edges.append((u, v))
                if not valid:
                    continue
                visited = set([0])
                stack = [0]
                while stack:
                    v = stack.pop()
                    for w in adj[v]:
                        if w not in visited:
                            visited.add(w)
                            stack.append(w)
                if len(visited) != n_try:
                    continue
                adj_list = [list(s) for s in adj]
                if _verify_girth_fast(adj_list, n_try, g):
                    return n_try, edges
            n_try += 64
    
    # Fallback: LUW construction
    q = 3
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    add, sub, mul = (lambda x,y:(x+y)%q), (lambda x,y:(x-y)%q), (lambda x,y:x*y%q)
    rule = {2:(1,1), 3:(2,1), 4:(1,2)}
    for c in range(5, d+1):
        r, c0 = (c-5)%4, c-(c-5)%4
        rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0]*(d-1)
            for c in range(2, d+1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side==0 else sub(x[c], mul(x[a], y[b]))
            yield 1-side, tuple(y)
    start = (0, (None,) + (0,)*d)
    index, order, edges = {start:0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
# EVOLVE-BLOCK-END
