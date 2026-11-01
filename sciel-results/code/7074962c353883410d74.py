"""Combined program: optimized voltage lifts + fast verification + LUW fallback + random search."""
# EVOLVE-BLOCK-START
import time as _time
import random as _random
from collections import deque

def _verify_girth_fast(adj, n, g):
    """Fast girth check via non-backtracking BFS from each vertex.
    Returns True if girth >= g, False otherwise."""
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

def _build_lift(base_edges, m, shifts):
    """Build a voltage graph lift over Z_m.
    base_edges: list of (u,v) on the base graph
    m: lift size
    shifts: list of voltage shifts, one per base edge
    Returns (n, adj_list, edge_list) or None if invalid."""
    n_base = max(max(u for u,v in base_edges), max(v for u,v in base_edges)) + 1
    n = n_base * m
    adj = [set() for _ in range(n)]
    edges = []
    for idx, (u, v) in enumerate(base_edges):
        vol = shifts[idx] % m
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
    adj_list = [list(s) for s in adj]
    return n, adj_list, edges

def _find_primitive_root(m):
    """Find a primitive root modulo m, or None if m is not prime."""
    if m < 2:
        return None
    if m == 2:
        return 1
    if m % 2 == 0:
        return None
    for i in range(3, int(m**0.5) + 1, 2):
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

def _primitive_root_shifts(m, e_count):
    """Generate e_count shifts as powers of a primitive root modulo m."""
    if m <= 1:
        return [0] * e_count
    g = _find_primitive_root(m)
    if g is None:
        shifts = []
        val = 1
        for i in range(e_count):
            shifts.append(val % m)
            val = (val * 2) % m
        return shifts
    shifts = []
    val = 1
    for i in range(e_count):
        shifts.append(val % m)
        val = (val * g) % m
    return shifts

def _luw_construct(k, g, d_override=None):
    """LUW construction with optional d override."""
    q = k
    if d_override is not None:
        d = d_override
    else:
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

def _random_lift_search(m, g, seed, num_trials=100):
    """Random voltage assignment search on Petersen lift."""
    rng = _random.Random(seed)
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    e_count = len(petersen_edges)
    
    for trial in range(num_trials):
        shifts = [rng.randint(0, m - 1) for _ in range(e_count)]
        result = _build_lift(petersen_edges, m, shifts)
        if result is not None:
            n_r, adj, edges = result
            if _verify_girth_fast(adj, n_r, g):
                return n_r, edges
    return None

def construct(k, g):
    start_time = _time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    if k != 3:
        n_luw, edges_luw = _luw_construct(k, g)
        return n_luw, edges_luw

    # k=3, target girth g (typically 16)
    if g % 2 == 0:
        moore = 2 * (2**(g//2) - 1)
    else:
        moore = 2 * (2**(g//2 - 1) - 1) + 2

    # Petersen graph (10 vertices, 15 edges)
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    e_count = len(petersen_edges)
    n_base = 10

    best_result = None
    best_n = float('inf')

    # Strategy: scan m from Moore bound upward, using primitive root shifts
    # Focus on primes for better algebraic structure
    m_min = max(2, (moore + n_base - 1) // n_base)
    
    # Build list of candidate m values, prioritizing primes
    candidates = []
    for m in range(m_min, m_min + 300):
        candidates.append(m)
    
    # Try each candidate m with primitive root shifts
    for m in candidates:
        if _time.time() > deadline - 5:
            break
        n = n_base * m
        if n >= best_n:
            continue
        
        # Primitive root shifts
        shifts = _primitive_root_shifts(m, e_count)
        result = _build_lift(petersen_edges, m, shifts)
        if result is not None:
            n_r, adj, edges = result
            if _verify_girth_fast(adj, n_r, g):
                best_n = n_r
                best_result = (n_r, edges)
                return n_r, edges
        
        # Second variant: negated primitive root shifts
        shifts2 = [(-s) % m for s in shifts]
        result = _build_lift(petersen_edges, m, shifts2)
        if result is not None:
            n_r, adj, edges = result
            if _verify_girth_fast(adj, n_r, g):
                best_n = n_r
                best_result = (n_r, edges)
                return n_r, edges
        
        # Third variant: shifted primitive root (multiply by a constant)
        if m > 1:
            shifts3 = [(s * 3) % m for s in shifts]
            result = _build_lift(petersen_edges, m, shifts3)
            if result is not None:
                n_r, adj, edges = result
                if _verify_girth_fast(adj, n_r, g):
                    best_n = n_r
                    best_result = (n_r, edges)
                    return n_r, edges

    # Random search phase: try random voltage assignments for promising m values
    if _time.time() < deadline - 10:
        for m in range(m_min, min(m_min + 50, m_min + 50)):
            if _time.time() > deadline - 5:
                break
            n = n_base * m
            if n >= best_n:
                continue
            result = _random_lift_search(m, g, seed=42 + m, num_trials=50)
            if result is not None:
                n_r, edges = result
                if n_r < best_n:
                    best_n = n_r
                    best_result = (n_r, edges)
                    return n_r, edges

    # Fallback: LUW with various d values
    if _time.time() < deadline - 5:
        for d_try in range(4, 16):
            if _time.time() > deadline - 3:
                break
            try:
                n_try, edges_try = _luw_construct(3, g, d_override=d_try)
                if n_try < best_n:
                    adj = [set() for _ in range(n_try)]
                    for u, v in edges_try:
                        adj[u].add(v)
                        adj[v].add(u)
                    adj_list = [list(s) for s in adj]
                    if _verify_girth_fast(adj_list, n_try, g):
                        best_n = n_try
                        best_result = (n_try, edges_try)
                        return n_try, edges_try
            except Exception:
                pass

    # Final fallback: standard LUW
    if best_result is not None:
        return best_result

    n_luw, edges_luw = _luw_construct(3, g)
    return n_luw, edges_luw
# EVOLVE-BLOCK-END
