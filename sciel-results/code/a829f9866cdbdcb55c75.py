"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import random as _random
from collections import deque

def _verify_girth_fast(adj, n, g):
    """Check if graph has girth >= g using non-backtracking BFS from each vertex.
    Returns True if girth >= g, False otherwise. Optimized with early termination."""
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
                    dq.append(state)
    return True

def _verify_girth_array(adj_list, n, g):
    """Faster verification using array-based distance tracking."""
    # adj_list: list of lists (sorted neighbor lists)
    # For each start vertex, do non-backtracking BFS
    # State: (vertex, prev_vertex) -> depth
    # Use dict but with early termination
    
    for start in range(n):
        # dist[v][prev] = depth, but we'll use a flat dict
        dist = {}
        dq = deque()
        dq.append((start, -1))
        dist[(start, -1)] = 0
        while dq:
            v, prev = dq.popleft()
            d = dist[(v, prev)]
            if d >= g - 1:
                continue
            for w in adj_list[v]:
                if w == prev:
                    continue
                if w == start and d + 1 >= 3:
                    return False
                state = (w, v)
                if state not in dist:
                    dist[state] = d + 1
                    dq.append(state)
    return True

def _luw_construct(q, g):
    """LUW construction for (q+1)-regular graphs."""
    if q == 3:
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
    elif q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add, sub, mul = (lambda x,y: x^y), (lambda x,y: x^y), (lambda x,y: table[x][y])
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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
    else:
        p = q
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        add, sub, mul = (lambda x,y:(x+y)%p), (lambda x,y:(x-y)%p), (lambda x,y:x*y%p)
        rule = {2:(1,1), 3:(2,1), 4:(1,2)}
        for c in range(5, d+1):
            r, c0 = (c-5)%4, c-(c-5)%4
            rule[c] = [(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r]
        def neighbours(vertex):
            side, x = vertex
            for first in range(p):
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

def _build_lift_petersen(m, voltages):
    """Build a voltage graph lift of the Petersen graph with given voltages.
    Returns (n, adj, edges) or None if invalid."""
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    
    n = 10 * m
    adj = [set() for _ in range(n)]
    edges = []
    
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
    
    return n, adj, edges

def _build_lift_k33(m, voltages):
    """Build a voltage graph lift of K3,3 with given voltages.
    K3,3 has 6 vertices, 9 edges. Lift has 6m vertices.
    Returns (n, adj, edges) or None if invalid."""
    k33_edges = [
        (0, 3), (0, 4), (0, 5),
        (1, 3), (1, 4), (1, 5),
        (2, 3), (2, 4), (2, 5)
    ]
    
    n = 6 * m
    adj = [set() for _ in range(n)]
    edges = []
    
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
    
    return n, adj, edges

def _find_primitive_root(m):
    """Find a primitive root modulo m, or return None."""
    if m <= 1:
        return None
    import math
    if m < 2:
        return None
    for i in range(2, int(math.isqrt(m)) + 1):
        if m % i == 0:
            return None
    phi = m - 1
    factors = []
    temp = phi
    d = 2
    while d * d <= temp:
        if temp % d == 0:
            factors.append(d)
            while temp % d == 0:
                temp //= d
        d += 1
    if temp > 1:
        factors.append(temp)
    
    for g in range(2, m):
        is_primitive = True
        for p in factors:
            if pow(g, phi // p, m) == 1:
                is_primitive = False
                break
        if is_primitive:
            return g
    return None

def _deterministic_voltage_lift_petersen(m, g):
    """Deterministic voltage lift of Petersen graph."""
    root = _find_primitive_root(m)
    if root is None:
        return None
    
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    
    voltages = {}
    for i, (u, v) in enumerate(petersen_edges):
        voltages[(u, v)] = (root ** i) % m
        voltages[(v, u)] = (-voltages[(u, v)]) % m
    
    result = _build_lift_petersen(m, voltages)
    if result is None:
        return None
    n, adj, edges = result
    if _verify_girth_fast(adj, n, g):
        return n, edges
    return None

def _deterministic_voltage_lift_k33(m, g):
    """Deterministic voltage lift of K3,3 graph."""
    root = _find_primitive_root(m)
    if root is None:
        return None
    
    k33_edges = [
        (0, 3), (0, 4), (0, 5),
        (1, 3), (1, 4), (1, 5),
        (2, 3), (2, 4), (2, 5)
    ]
    
    voltages = {}
    for i, (u, v) in enumerate(k33_edges):
        voltages[(u, v)] = (root ** i) % m
        voltages[(v, u)] = (-voltages[(u, v)]) % m
    
    result = _build_lift_k33(m, voltages)
    if result is None:
        return None
    n, adj, edges = result
    if _verify_girth_fast(adj, n, g):
        return n, edges
    return None

def _random_voltage_lift_petersen(m, g, seed):
    """Random voltage assignment on Petersen lift."""
    rng = _random.Random(seed)
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    
    voltages = {}
    for u, v in petersen_edges:
        voltages[(u, v)] = rng.randint(0, m - 1)
        voltages[(v, u)] = (-voltages[(u, v)]) % m
    
    result = _build_lift_petersen(m, voltages)
    if result is None:
        return None
    n, adj, edges = result
    if _verify_girth_fast(adj, n, g):
        return n, edges
    return None

def _random_voltage_lift_k33(m, g, seed):
    """Random voltage assignment on K3,3 lift."""
    rng = _random.Random(seed)
    k33_edges = [
        (0, 3), (0, 4), (0, 5),
        (1, 3), (1, 4), (1, 5),
        (2, 3), (2, 4), (2, 5)
    ]
    
    voltages = {}
    for u, v in k33_edges:
        voltages[(u, v)] = rng.randint(0, m - 1)
        voltages[(v, u)] = (-voltages[(u, v)]) % m
    
    result = _build_lift_k33(m, voltages)
    if result is None:
        return None
    n, adj, edges = result
    if _verify_girth_fast(adj, n, g):
        return n, edges
    return None

def _random_3reg(n, g, seed):
    """Generate a random 3-regular graph and check girth."""
    rng = _random.Random(seed)
    stubs = []
    for v in range(n):
        for _ in range(3):
            stubs.append(v)
    rng.shuffle(stubs)
    adj = [set() for _ in range(n)]
    edges = []
    valid = True
    for i in range(0, n * 3, 2):
        u, v = stubs[i], stubs[i + 1]
        if u == v or v in adj[u]:
            valid = False
            break
        adj[u].add(v)
        adj[v].add(u)
        edges.append((u, v))
    if not valid:
        return None
    visited = set([0])
    stack = [0]
    while stack:
        v = stack.pop()
        for w in adj[v]:
            if w not in visited:
                visited.add(w)
                stack.append(w)
    if len(visited) != n:
        return None
    if _verify_girth_fast(adj, n, g):
        return n, edges
    return None

def _greedy_tree_extension(g, max_n, seed):
    """Randomized greedy tree-extension algorithm.
    Builds a 3-regular graph by extending a tree structure,
    trying to close cycles of length >= g."""
    rng = _random.Random(seed)
    
    # Start with a path of length g (to ensure girth >= g initially)
    # Actually, let's build incrementally
    # We'll maintain a list of vertices and their edges
    # Each vertex can have up to 3 edges
    # We try to extend the graph while maintaining girth >= g
    
    # This is complex; for now, fall back to random construction
    # But let's try a different approach: build a tree, then add edges
    # that create cycles of length >= g
    
    # For g=16, we need to be very careful
    # Let's try constructing a 3-regular graph with specific structure
    
    # Actually, let's use a different strategy: 
    # Build the graph as a "cage" using known constructions
    # For (3,16), the best known is 936 vertices
    
    # Let's try to construct using the known 936-vertex graph structure
    # This is a specific known construction
    
    # For now, return None to use other strategies
    return None

def construct(k, g):
    start_time = _time.time()
    deadline = start_time + 115
    
    if k != 3:
        q = k
        n, edges = _luw_construct(q, g)
        return n, edges
    
    # Moore bound for k=3, girth g
    if g % 2 == 0:
        moore = 2 * (2**(g//2) - 1)
    else:
        moore = 2 * (2**(g//2 - 1) - 1) + 2
    
    best_result = None
    best_n = float('inf')
    
    # Strategy 1: Try voltage lifts with both Petersen and K3,3 bases
    # K3,3 lifts produce graphs of size 6m, Petersen lifts produce 10m
    # For the same n, K3,3 allows smaller m values
    
    # For Petersen: n = 10m, so m = n/10
    # For K3,3: n = 6m, so m = n/6
    
    # Start from the Moore bound
    m_min_petersen = max(1, (moore + 9) // 10)
    m_min_k33 = max(1, (moore + 5) // 6)
    
    # Try K3,3 lifts first (smaller m values for same n)
    for m in range(m_min_k33, m_min_k33 + 300):
        if _time.time() > deadline - 10:
            break
        n = 6 * m
        if n < moore:
            continue
        if n >= best_n:
            break
        
        # Try deterministic primitive root voltage
        result = _deterministic_voltage_lift_k33(m, g)
        if result is not None:
            n_lift, edges = result
            if n_lift < best_n:
                best_n = n_lift
                best_result = (n_lift, edges)
            break
        
        # Try random voltage assignments
        for attempt in range(30):
            if _time.time() > deadline - 10:
                break
            result = _random_voltage_lift_k33(m, g, seed=attempt * 7919 + m * 13)
            if result is not None:
                n_lift, edges = result
                if n_lift < best_n:
                    best_n = n_lift
                    best_result = (n_lift, edges)
                break
    
    # Try Petersen lifts
    for m in range(m_min_petersen, m_min_petersen + 200):
        if _time.time() > deadline - 10:
            break
        n = 10 * m
        if n < moore:
            continue
        if n >= best_n:
            break
        
        result = _deterministic_voltage_lift_petersen(m, g)
        if result is not None:
            n_lift, edges = result
            if n_lift < best_n:
                best_n = n_lift
                best_result = (n_lift, edges)
            break
        
        for attempt in range(20):
            if _time.time() > deadline - 10:
                break
            result = _random_voltage_lift_petersen(m, g, seed=attempt * 7919 + m * 13)
            if result is not None:
                n_lift, edges = result
                if n_lift < best_n:
                    best_n = n_lift
                    best_result = (n_lift, edges)
                break
    
    # Strategy 2: Try LUW construction as a strong baseline
    if _time.time() < deadline - 15:
        n_luw, edges_luw = _luw_construct(3, g)
        if n_luw < best_n:
            best_n = n_luw
            best_result = (n_luw, edges_luw)
    
    # Strategy 3: Random 3-regular graphs near the Moore bound
    if _time.time() < deadline - 15 and best_result is None:
        n_try = moore
        if n_try % 2 != 0:
            n_try += 1
        while n_try < 5000 and _time.time() < deadline - 10:
            for attempt in range(30):
                if _time.time() > deadline - 10:
                    break
                result = _random_3reg(n_try, g, seed=attempt * 104729 + n_try)
                if result is not None:
                    n_r, edges_r = result
                    if n_r < best_n:
                        best_n = n_r
                        best_result = (n_r, edges_r)
                    break
            n_try += 32
    
    # Strategy 4: More extensive random search
    if _time.time() < deadline - 10 and best_result is None:
        for n_try in range(moore, moore + 200, 2):
            if _time.time() > deadline - 10:
                break
            for attempt in range(100):
                if _time.time() > deadline - 5:
                    break
                result = _random_3reg(n_try, g, seed=attempt * 31 + n_try * 17)
                if result is not None:
                    best_result = result
                    break
            if best_result is not None:
                break
    
    # Fallback: LUW construction
    if best_result is None:
        n_luw, edges_luw = _luw_construct(3, g)
        best_result = (n_luw, edges_luw)
    
    return best_result
# EVOLVE-BLOCK-END
