"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import random as _random
from collections import deque

def _verify_girth(adj, n, g):
    """Check if graph has girth >= g using non-backtracking BFS from each vertex.
    Returns True if girth >= g, False otherwise."""
    for start in range(n):
        # State: (vertex, prev_vertex), track min distance
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
                    return False  # found cycle shorter than g
                state = (w, v)
                if state not in dist:
                    dist[state] = d + 1
                    dq.append((w, v))
    return True

def _build_lift(m, seed=0):
    """Build a voltage graph lift of the Petersen graph over Z_m.
    Returns (n, adj, edges) or None if invalid."""
    rng = _random.Random(seed)
    
    # Petersen graph: 10 vertices, 15 edges, 3-regular, girth 5
    # Vertices 0-4 form outer 5-cycle, 5-9 form inner star
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # outer cycle
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),  # inner star
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)   # spokes
    ]
    
    n = 10 * m
    adj = [set() for _ in range(n)]
    edges = []
    
    # Assign random voltages to each edge
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
    
    # Check degree
    for i in range(n):
        if len(adj[i]) != 3:
            return None
    
    # Check no multi-edges (should be fine with sets, but verify)
    for i in range(n):
        if len(adj[i]) != len(adj[i]):
            return None
    
    return n, adj, edges

def construct(k, g):
    start_time = _time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
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
    
    # For k=3: Use voltage graph lift of Petersen graph
    # Moore bound for k=3, girth g: 2*(2^(g/2)-1) for even g
    if g % 2 == 0:
        moore = 2 * (2**(g//2) - 1)
    else:
        moore = 2 * (2**(g//2 - 1) - 1) + 2
    
    best_result = None
    
    # Try different lift sizes m
    # n = 10*m, need n >= moore, so m >= moore/10
    m_min = max(1, (moore + 9) // 10)
    
    # Also try some smaller m values in case the base graph structure helps
    m_candidates = list(range(max(1, m_min - 5), m_min + 200))
    # Remove duplicates and sort
    m_candidates = sorted(set(m for m in m_candidates if m >= 1))
    
    for m in m_candidates:
        if _time.time() > deadline - 5:
            break
        n = 10 * m
        if n < moore:
            continue
        
        # Try multiple random voltage assignments
        max_attempts = 50
        for attempt in range(max_attempts):
            if _time.time() > deadline - 3:
                break
            result = _build_lift(m, seed=attempt)
            if result is None:
                continue
            n, adj, edges = result
            if _verify_girth(adj, n, g):
                best_result = (n, edges)
                return n, edges
    
    # If no lift worked, try random 3-regular graphs
    if _time.time() < deadline - 10:
        n_try = moore
        while n_try < 10000 and _time.time() < deadline - 10:
            if n_try % 2 != 0:
                n_try += 1
                continue
            for attempt in range(50):
                if _time.time() > deadline - 5:
                    break
                # Generate random 3-regular graph
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
                # Check connectedness
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
                if _verify_girth(adj, n_try, g):
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
