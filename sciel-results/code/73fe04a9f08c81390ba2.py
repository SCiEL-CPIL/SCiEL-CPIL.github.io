"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import random as _random
from collections import deque

def _verify_girth(adj_list, n, g):
    """Check if graph has girth >= g using non-backtracking BFS from each vertex.
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
    """LUW construction for (q+1)-regular graphs. Returns (n, edges)."""
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

def _build_lift(base_edges, n_base, m, voltages):
    """Build a voltage graph lift. Returns (n, adj_list, edges) or None if invalid."""
    n = n_base * m
    adj = [set() for _ in range(n)]
    edges = []
    
    for u, v in base_edges:
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
    
    adj_list = [sorted(adj[i]) for i in range(n)]
    return n, adj_list, edges

def _find_primitive_root(m):
    """Find a primitive root modulo m (requires m prime). Returns None otherwise."""
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

def _is_prime(n):
    """Simple primality test."""
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _petersen_edges():
    return [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]

def _k33_edges():
    return [
        (0, 3), (0, 4), (0, 5),
        (1, 3), (1, 4), (1, 5),
        (2, 3), (2, 4), (2, 5)
    ]

def _voltage_lift_petersen(m, g, seed, n_attempts=50):
    """Try voltage assignments on Petersen graph lift.
    Returns (n, edges) if valid, else None."""
    p_edges = _petersen_edges()
    n_base = 10
    rng = _random.Random(seed)
    
    # First try deterministic: powers of primitive root
    root = _find_primitive_root(m)
    if root is not None:
        voltages = {}
        for i, (u, v) in enumerate(p_edges):
            voltages[(u, v)] = (root ** i) % m
            voltages[(v, u)] = (-voltages[(u, v)]) % m
        result = _build_lift(p_edges, n_base, m, voltages)
        if result is not None:
            n, adj_list, edges = result
            if _verify_girth(adj_list, n, g):
                return n, edges
    
    # Try structured: use small set of voltage values
    for attempt in range(n_attempts):
        voltages = {}
        if attempt < 10:
            # Linear voltages
            base_val = rng.randint(1, m - 1)
            for i, (u, v) in enumerate(p_edges):
                vol = (base_val * (i + 1)) % m
                voltages[(u, v)] = vol
                voltages[(v, u)] = (-vol) % m
        elif attempt < 20:
            # Quadratic voltages
            base_val = rng.randint(1, m - 1)
            for i, (u, v) in enumerate(p_edges):
                vol = (base_val * (i + 1) * (i + 1)) % m
                voltages[(u, v)] = vol
                voltages[(v, u)] = (-vol) % m
        else:
            # Random
            for u, v in p_edges:
                vol = rng.randint(0, m - 1)
                voltages[(u, v)] = vol
                voltages[(v, u)] = (-vol) % m
        
        result = _build_lift(p_edges, n_base, m, voltages)
        if result is None:
            continue
        n, adj_list, edges = result
        if _verify_girth(adj_list, n, g):
            return n, edges
    
    return None

def _voltage_lift_k33(m, g, seed, n_attempts=50):
    """Try voltage assignments on K3,3 graph lift.
    Returns (n, edges) if valid, else None."""
    k_edges = _k33_edges()
    n_base = 6
    rng = _random.Random(seed)
    
    root = _find_primitive_root(m)
    if root is not None:
        voltages = {}
        for i, (u, v) in enumerate(k_edges):
            voltages[(u, v)] = (root ** i) % m
            voltages[(v, u)] = (-voltages[(u, v)]) % m
        result = _build_lift(k_edges, n_base, m, voltages)
        if result is not None:
            n, adj_list, edges = result
            if _verify_girth(adj_list, n, g):
                return n, edges
    
    for attempt in range(n_attempts):
        voltages = {}
        if attempt < 10:
            base_val = rng.randint(1, m - 1)
            for i, (u, v) in enumerate(k_edges):
                vol = (base_val * (i + 1)) % m
                voltages[(u, v)] = vol
                voltages[(v, u)] = (-vol) % m
        elif attempt < 20:
            base_val = rng.randint(1, m - 1)
            for i, (u, v) in enumerate(k_edges):
                vol = (base_val * (i + 1) * (i + 1)) % m
                voltages[(u, v)] = vol
                voltages[(v, u)] = (-vol) % m
        else:
            for u, v in k_edges:
                vol = rng.randint(0, m - 1)
                voltages[(u, v)] = vol
                voltages[(v, u)] = (-vol) % m
        
        result = _build_lift(k_edges, n_base, m, voltages)
        if result is None:
            continue
        n, adj_list, edges = result
        if _verify_girth(adj_list, n, g):
            return n, edges
    
    return None

def _search_prime_lifts(g, deadline, best_n):
    """Search for good prime lifts of Petersen and K3,3.
    Returns (n, edges) for best found, or None."""
    p_edges = _petersen_edges()
    k_edges = _k33_edges()
    
    # For girth g, we need m large enough. Moore bound for 3-reg, girth g:
    if g % 2 == 0:
        moore = 2 * (2**(g//2) - 1)
    else:
        moore = 2 * (2**(g//2 - 1) - 1) + 2
    
    # n_lift = n_base * m, need n_lift >= moore
    # For Petersen: m >= moore/10
    # For K3,3: m >= moore/6
    
    m_min_petersen = max(2, (moore + 9) // 10)
    m_min_k33 = max(2, (moore + 5) // 6)
    
    best_result = None
    best_found = best_n
    
    # Try primes in increasing order for K3,3 (smaller n_base = smaller total n)
    m = m_min_k33
    while m < best_found // 6 and _time.time() < deadline - 30:
        if _is_prime(m):
            result = _voltage_lift_k33(m, g, seed=m * 1000003, n_attempts=30)
            if result is not None:
                n_r, edges_r = result
                if n_r < best_found:
                    best_found = n_r
                    best_result = (n_r, edges_r)
        m += 1
    
    # Try primes for Petersen
    m = m_min_petersen
    while m < best_found // 10 and _time.time() < deadline - 25:
        if _is_prime(m):
            result = _voltage_lift_petersen(m, g, seed=m * 1000003, n_attempts=30)
            if result is not None:
                n_r, edges_r = result
                if n_r < best_found:
                    best_found = n_r
                    best_result = (n_r, edges_r)
        m += 1
    
    return best_result

def _composite_lift_search(g, deadline, best_n):
    """Search for good composite lifts (not just primes).
    Returns (n, edges) for best found, or None."""
    p_edges = _petersen_edges()
    k_edges = _k33_edges()
    
    if g % 2 == 0:
        moore = 2 * (2**(g//2) - 1)
    else:
        moore = 2 * (2**(g//2 - 1) - 1) + 2
    
    best_result = None
    best_found = best_n
    
    # Try K3,3 lifts with various m values
    m_min = max(2, (moore + 5) // 6)
    m = m_min
    while m < best_found // 6 and _time.time() < deadline - 20:
        # Skip if we already tried this as prime
        if not _is_prime(m):
            result = _voltage_lift_k33(m, g, seed=m * 7919, n_attempts=20)
            if result is not None:
                n_r, edges_r = result
                if n_r < best_found:
                    best_found = n_r
                    best_result = (n_r, edges_r)
        m += 1
    
    return best_result

def construct(k, g):
    start_time = _time.time()
    deadline = start_time + 115
    
    if k != 3:
        n, edges = _luw_construct(k, g)
        return n, edges
    
    # Moore bound for k=3, girth g
    if g % 2 == 0:
        moore = 2 * (2**(g//2) - 1)
    else:
        moore = 2 * (2**(g//2 - 1) - 1) + 2
    
    best_result = None
    best_n = float('inf')
    
    # Strategy 1: LUW construction (deterministic, fast)
    n_luw, edges_luw = _luw_construct(3, g)
    best_n = n_luw
    best_result = (n_luw, edges_luw)
    
    # Strategy 2: Prime voltage lifts (structural approach from inspiration)
    result = _search_prime_lifts(g, deadline, best_n)
    if result is not None:
        n_r, edges_r = result
        if n_r < best_n:
            best_n = n_r
            best_result = (n_r, edges_r)
    
    # Strategy 3: Composite voltage lifts
    if _time.time() < deadline - 15:
        result = _composite_lift_search(g, deadline, best_n)
        if result is not None:
            n_r, edges_r = result
            if n_r < best_n:
                best_n = n_r
                best_result = (n_r, edges_r)
    
    # Strategy 4: If we have time, try more random attempts on the best base
    if _time.time() < deadline - 10 and best_result is not None:
        # Try additional random voltage assignments on K3,3 with m near moore/6
        m_target = max(2, (moore + 5) // 6)
        for offset in range(0, 50):
            m = m_target + offset
            if m * 6 >= best_n:
                break
            if _time.time() > deadline - 8:
                break
            result = _voltage_lift_k33(m, g, seed=m * 314159 + offset, n_attempts=10)
            if result is not None:
                n_r, edges_r = result
                if n_r < best_n:
                    best_n = n_r
                    best_result = (n_r, edges_r)
    
    return best_result
# EVOLVE-BLOCK-END
