"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time as _time
import random as _random
from collections import deque

def _verify_girth_array(adj_list, n, g):
    """Faster verification using array-based distance tracking with flat index."""
    # Precompute neighbor position lookup: for vertex v, adj_pos[v][w] = index of w in adj_list[v]
    adj_pos = [{} for _ in range(n)]
    for v in range(n):
        for i, w in enumerate(adj_list[v]):
            adj_pos[v][w] = i
    
    INF = 10**9
    for start in range(n):
        dist_arr = [INF] * (n * 4)
        # slot 0 = came from nowhere (start), slots 1-3 = came from neighbor idx 0,1,2
        dist_arr[start * 4 + 0] = 0
        head = 0
        tail = 1
        # Pre-allocate queue arrays once
        qv = [0] * (n * 4 + 1)
        qs = [0] * (n * 4 + 1)
        qd = [0] * (n * 4 + 1)
        qv[0] = start
        qs[0] = 0
        qd[0] = 0
        while head < tail:
            v = qv[head]
            slot = qs[head]
            d = qd[head]
            head += 1
            if d >= g - 1:
                continue
            prev_idx = slot - 1  # -1 for slot 0, else 0,1,2
            for i, w in enumerate(adj_list[v]):
                if i == prev_idx:
                    continue
                if w == start and d + 1 >= 3:
                    return False
                # At vertex w, the prev neighbor is v, which is at position adj_pos[w][v]
                w_prev_idx = adj_pos[w][v]
                w_slot = w_prev_idx + 1
                idx = w * 4 + w_slot
                if dist_arr[idx] > d + 1:
                    dist_arr[idx] = d + 1
                    qv[tail] = w
                    qs[tail] = w_slot
                    qd[tail] = d + 1
                    tail += 1
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

def _get_base_graphs():
    """Return a list of base 3-regular graphs as (edges, n_base)."""
    # Petersen graph
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    # K_{3,3}
    k33_edges = [
        (0, 3), (0, 4), (0, 5),
        (1, 3), (1, 4), (1, 5),
        (2, 3), (2, 4), (2, 5)
    ]
    # Also try a 6-vertex triangle-free 3-reg: two triangles with cross edges
    # Actually K3,3 is the only 3-reg on 6 vertices that's simple.
    # Let's also add the utility graph (K3,3) and Petersen
    return [(petersen_edges, 10), (k33_edges, 6)]

def _build_lift(base_edges, n_base, m, voltages, g):
    """Build a voltage graph lift. Returns (n, adj_list, edges) or None if invalid."""
    n = n_base * m
    adj = [[] for _ in range(n)]
    edges_set = set()
    edges = []
    
    for u, v in base_edges:
        vol = voltages[(u, v)]
        for i in range(m):
            a = u * m + i
            b = v * m + (i + vol) % m
            if a == b:
                return None
            key = (min(a, b), max(a, b))
            if key in edges_set:
                continue
            edges_set.add(key)
            adj[a].append(b)
            adj[b].append(a)
            edges.append((a, b))
    
    for i in range(n):
        if len(adj[i]) != 3:
            return None
    
    adj_list = [sorted(adj[i]) for i in range(n)]
    return n, adj_list, edges

def _build_directed_lift(base_edges, n_base, m, directed_voltages, g):
    """Build a directed voltage graph lift.
    directed_voltages: dict mapping (u,v) -> vol for each directed edge in base.
    The base graph edges are treated as undirected; both directions must be specified.
    """
    n = n_base * m
    # For 3-regular: each vertex has exactly 3 neighbors
    # Build adjacency directly as lists
    adj = [[] for _ in range(n)]
    edges_set = set()
    edges = []
    
    for (u, v), vol in directed_voltages.items():
        for i in range(m):
            a = u * m + i
            b = v * m + (i + vol) % m
            if a == b:
                return None  # loop
            key = (min(a, b), max(a, b))
            if key in edges_set:
                continue
            edges_set.add(key)
            adj[a].append(b)
            adj[b].append(a)
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

def _voltage_lift_search(base_edges, n_base, m, g, seed, n_attempts=50):
    """Try multiple random voltage assignments on a base graph lift.
    Returns (n, edges) if a valid girth-g graph is found, else None."""
    rng = _random.Random(seed)
    n_lift = n_base * m
    
    for attempt in range(n_attempts):
        voltages = {}
        for u, v in base_edges:
            vol = rng.randint(0, m - 1)
            voltages[(u, v)] = vol
            voltages[(v, u)] = (-vol) % m
        
        result = _build_lift(base_edges, n_base, m, voltages, g)
        if result is None:
            continue
        n, adj_list, edges = result
        if _verify_girth_array(adj_list, n, g):
            return n, edges
    
    return None

def _directed_voltage_search(base_edges, n_base, m, g, seed, n_attempts=200):
    """Try random directed voltage assignments (independent for each direction).
    This is the standard approach for cage graphs."""
    rng = _random.Random(seed)
    n_lift = n_base * m
    
    for attempt in range(n_attempts):
        directed_voltages = {}
        for u, v in base_edges:
            vol_uv = rng.randint(0, m - 1)
            vol_vu = rng.randint(0, m - 1)
            directed_voltages[(u, v)] = vol_uv
            directed_voltages[(v, u)] = vol_vu
        
        result = _build_directed_lift(base_edges, n_base, m, directed_voltages, g)
        if result is None:
            continue
        n, adj_list, edges = result
        if _verify_girth_array(adj_list, n, g):
            return n, edges
    
    return None

def _primitive_root_directed_lift(base_edges, n_base, m, g):
    """Use primitive root powers for directed voltage assignment.
    Assign voltage root^i to edge i in one direction and root^j to reverse.
    For undirected base edges, assign different powers to each direction."""
    root = _find_primitive_root(m)
    if root is None:
        return None
    
    n_edges = len(base_edges)
    # Assign distinct powers to each directed edge
    # For edge i: forward = root^(2i), backward = root^(2i+1)
    # This ensures all 2*n_edges voltages are distinct (if m is prime and root is primitive)
    directed_voltages = {}
    for i, (u, v) in enumerate(base_edges):
        vol_uv = (root ** (2 * i)) % m
        vol_vu = (root ** (2 * i + 1)) % m
        directed_voltages[(u, v)] = vol_uv
        directed_voltages[(v, u)] = vol_vu
    
    result = _build_directed_lift(base_edges, n_base, m, directed_voltages, g)
    if result is None:
        return None
    n, adj_list, edges = result
    if _verify_girth_array(adj_list, n, g):
        return n, edges
    return None

def _deterministic_voltage_lift(base_edges, n_base, m, g):
    """Use deterministic voltage assignment based on primitive root powers."""
    root = _find_primitive_root(m)
    if root is None:
        return None
    
    voltages = {}
    for i, (u, v) in enumerate(base_edges):
        voltages[(u, v)] = (root ** i) % m
        voltages[(v, u)] = (-voltages[(u, v)]) % m
    
    result = _build_lift(base_edges, n_base, m, voltages, g)
    if result is None:
        return None
    n, adj_list, edges = result
    if _verify_girth_array(adj_list, n, g):
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
    adj_list = [sorted(adj[i]) for i in range(n)]
    if _verify_girth_array(adj_list, n, g):
        return n, edges
    return None

def _structured_voltage_lift(base_edges, n_base, m, g, seed):
    """Try structured voltage assignments: quadratic residues, linear, etc."""
    rng = _random.Random(seed)
    n_lift = n_base * m
    n_edges = len(base_edges)
    
    # Strategy: use powers of a base element (undirected)
    for base_elem in range(2, min(m, 20) + 1):
        voltages = {}
        for i, (u, v) in enumerate(base_edges):
            vol = (base_elem ** i) % m
            voltages[(u, v)] = vol
            voltages[(v, u)] = (-vol) % m
        result = _build_lift(base_edges, n_base, m, voltages, g)
        if result is None:
            continue
        n, adj_list, edges = result
        if _verify_girth_array(adj_list, n, g):
            return n, edges
    
    # Strategy: random but with structure (each edge gets voltage from a small set)
    for attempt in range(30):
        voltages = {}
        # Use a random permutation of 0..m-1 for the voltages
        perms = list(range(m))
        rng.shuffle(perms)
        for i, (u, v) in enumerate(base_edges):
            vol = perms[i % m]
            voltages[(u, v)] = vol
            voltages[(v, u)] = (-vol) % m
        result = _build_lift(base_edges, n_base, m, voltages, g)
        if result is None:
            continue
        n, adj_list, edges = result
        if _verify_girth_array(adj_list, n, g):
            return n, edges
    
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
    
    # Get base graphs
    base_graphs = _get_base_graphs()
    
    # Strategy 1: Directed voltage lifts - the primary approach for cage graphs
    # For each base graph, try increasing m with directed voltage assignments
    for base_edges, n_base in base_graphs:
        if _time.time() > deadline - 30:
            break
        m_min = max(1, (moore + n_base - 1) // n_base)
        m_max = m_min + 800
        
        for m in range(m_min, m_max):
            if _time.time() > deadline - 25:
                break
            n_lift = n_base * m
            if n_lift < moore:
                continue
            if n_lift >= best_n:
                break
            
            # Try primitive root directed lift first (deterministic, fast)
            if m > 1:
                result = _primitive_root_directed_lift(base_edges, n_base, m, g)
                if result is not None:
                    n_r, edges_r = result
                    if n_r < best_n:
                        best_n = n_r
                        best_result = (n_r, edges_r)
                    break
            
            # Try directed random voltage assignments (more attempts with faster verify)
            n_attempts = 300 if m <= 200 else 150
            result = _directed_voltage_search(base_edges, n_base, m, g, seed=m * 100003 + n_base * 7919, n_attempts=n_attempts)
            if result is not None:
                n_r, edges_r = result
                if n_r < best_n:
                    best_n = n_r
                    best_result = (n_r, edges_r)
                break
            
            # Try undirected random voltage as fallback
            result = _voltage_lift_search(base_edges, n_base, m, g, seed=m * 100000 + n_base, n_attempts=100)
            if result is not None:
                n_r, edges_r = result
                if n_r < best_n:
                    best_n = n_r
                    best_result = (n_r, edges_r)
                break
    
    # Strategy 2: LUW construction as fallback
    if _time.time() < deadline - 15:
        n_luw, edges_luw = _luw_construct(3, g)
        if n_luw < best_n:
            best_n = n_luw
            best_result = (n_luw, edges_luw)
    
    # Strategy 3: If we have time, try more m values with the best base graph
    if best_result is not None and _time.time() < deadline - 10:
        # Continue searching with more m values for the first base graph
        base_edges, n_base = base_graphs[0]
        m_min = max(1, (moore + n_base - 1) // n_base)
        m_start = m_min + 800
        m_end = min(m_start + 400, best_n // n_base)
        for m in range(m_start, m_end):
            if _time.time() > deadline - 8:
                break
            n_lift = n_base * m
            if n_lift >= best_n:
                break
            n_attempts = 100
            result = _directed_voltage_search(base_edges, n_base, m, g, seed=m * 99991 + 5, n_attempts=n_attempts)
            if result is not None:
                n_r, edges_r = result
                if n_r < best_n:
                    best_n = n_r
                    best_result = (n_r, edges_r)
                break
    
    # Fallback: LUW construction
    if best_result is None:
        n_luw, edges_luw = _luw_construct(3, g)
        best_result = (n_luw, edges_luw)
    
    return best_result
# EVOLVE-BLOCK-END
