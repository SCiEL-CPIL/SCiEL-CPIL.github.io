"""Voltage graph construction for 3-regular girth-16 graph using PSL(2,p)."""
import time
import random
from collections import deque

def construct(k, g):
    """Construct a 3-regular graph with girth >= g using voltage graph on PSL(2,p)."""
    if k != 3:
        return _luw_fallback(k, g)
    
    start_time = time.time()
    deadline = start_time + 130.0
    
    # Base graph: Petersen (10 vertices, 15 edges)
    base_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    n_base = 10
    base_adj = [[] for _ in range(n_base)]
    for u, v in base_edges:
        base_adj[u].append(v)
        base_adj[v].append(u)
    
    # Precompute all non-backtracking closed walks of length 3 to g-1
    # Represented as sequences of directed edge indices
    edge_dir = {}  # (u,v) -> index, (v,u) -> index+n_edges
    n_edges = len(base_edges)
    for i, (u, v) in enumerate(base_edges):
        edge_dir[(u, v)] = i
        edge_dir[(v, u)] = i + n_edges
    
    # Generate closed non-backtracking walks
    closed_walks = []  # list of (length, [directed_edge_indices])
    for sv in range(n_base):
        # DFS: state = (current, prev_vertex, length, path)
        def dfs_walk(cur, prev, length, path):
            if length >= g - 1:
                return
            for nv in base_adj[cur]:
                if nv == prev:
                    continue
                new_path = path + [edge_dir[(cur, nv)]]
                if nv == sv and len(new_path) >= 3:
                    closed_walks.append((len(new_path), new_path))
                dfs_walk(nv, cur, length + 1, new_path)
        
        for nv in base_adj[sv]:
            dfs_walk(nv, sv, 1, [edge_dir[(sv, nv)]])
    
    # Deduplicate: a walk and its reverse/rotation represent same cycle
    # For voltage check, we need to check each distinct cycle
    # Use a set of frozensets of edge indices (undirected) for dedup
    seen_cycles = set()
    unique_walks = []
    for length, path in closed_walks:
        # Get undirected edge set
        undirected = frozenset(e % n_edges for e in path)
        # Also need to check orientation matters for voltage
        # But for girth, if a cycle has identity voltage in one direction,
        # it also has identity in the other (inverse of identity is identity)
        if undirected not in seen_cycles:
            seen_cycles.add(undirected)
            unique_walks.append((length, path))
    
    # PSL(2,p) operations
    def mat_mul(m1, m2, p):
        a1, b1, c1, d1 = m1
        a2, b2, c2, d2 = m2
        return ((a1*a2 + b1*c2) % p, (a1*b2 + b1*d2) % p,
                (c1*a2 + d1*c2) % p, (c1*b2 + d1*d2) % p)
    
    def mat_inv(m, p):
        a, b, c, d = m
        return (d % p, (-b) % p, (-c) % p, a % p)
    
    def mat_is_identity(m, p):
        a, b, c, d = m
        return (a == 1 and b == 0 and c == 0 and d == 1) or \
               (a == p-1 and b == 0 and c == 0 and d == p-1)
    
    def mat_identity(p):
        return (1, 0, 0, 1)
    
    # Efficient random PSL(2,p) element generation
    def random_psl_element(p, rng):
        # Pick a,b,c freely, then d = (1 + b*c) / a mod p if a != 0
        # If a == 0, need -b*c = 1, so b*c = p-1
        while True:
            a = rng.randint(1, p - 1)  # a != 0
            b = rng.randint(0, p - 1)
            c = rng.randint(0, p - 1)
            # d = (1 + b*c) * a^{-1} mod p
            d = (1 + b * c) * pow(a, p - 2, p) % p
            return (a, b, c, d)
    
    # Check if voltage assignment gives girth >= g
    def check_girth_fast(voltages, p):
        """Check no closed walk of length < g has identity voltage product."""
        # Build directed voltage lookup
        # voltages: dict (u,v) -> matrix for u<v in base_edges
        # For directed edge index e: if e < n_edges, it's (u,v) forward
        # if e >= n_edges, it's (v,u) = inverse of (u,v)
        dir_volt = [None] * (2 * n_edges)
        for i, (u, v) in enumerate(base_edges):
            m = voltages[(u, v)]
            dir_volt[i] = m
            dir_volt[i + n_edges] = mat_inv(m, p)
        
        for length, path in unique_walks:
            if length >= g:
                break
            # Compute product of voltages along the walk
            product = mat_identity(p)
            for e_idx in path:
                product = mat_mul(product, dir_volt[e_idx], p)
                # Early termination: if product becomes identity before end,
                # we still need to continue (it might not be closed yet)
            if mat_is_identity(product, p):
                return False
        return True
    
    # Try different primes
    best_n = None
    best_edges = None
    
    # Moore bound for k=3, g=16: 2 * sum_{i=0}^{6} 2^i = 2*(2^7-1) = 254
    # But actual lower bound is 512. Try primes giving n = 10*p
    # For girth 16, we need p large enough
    # Start from smaller primes and work up
    
    primes = []
    for p in range(53, 500):
        if p % 2 == 0 and p != 2:
            continue
        is_prime = True
        for d in range(3, int(p**0.5) + 1, 2):
            if p % d == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(p)
    
    for p in primes:
        if time.time() > deadline - 15:
            break
        
        n = 10 * p
        if best_n is not None and n >= best_n:
            break
        
        # Try random voltage assignments
        rng = random.Random(12345 + p)
        max_attempts = 50000
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Generate random voltages for the 15 edges
            voltages = {}
            for (u, v) in base_edges:
                voltages[(u, v)] = random_psl_element(p, rng)
            
            # Quick filter: no edge voltage should be identity or order-2
            valid = True
            for (u, v) in base_edges:
                m = voltages[(u, v)]
                if mat_is_identity(m, p):
                    valid = False
                    break
                # Check order 2: m^2 = identity
                m2 = mat_mul(m, m, p)
                if mat_is_identity(m2, p):
                    valid = False
                    break
            if not valid:
                continue
            
            # Full girth check
            if check_girth_fast(voltages, p):
                # Build the lift graph
                edges = _build_lift(base_edges, base_adj, voltages, p)
                if edges is not None:
                    n_vertices = len(edges) * 2 // 3
                    if best_n is None or n_vertices < best_n:
                        best_n = n_vertices
                        best_edges = edges
                        break  # Found valid for this p, try smaller p
    
    if best_edges is not None:
        return best_n, best_edges
    
    return _luw_fallback(k, g)

def _build_lift(base_edges, base_adj, voltages, p):
    """Build the voltage lift graph. Returns list of edges or None if disconnected."""
    n_edges = len(base_edges)
    
    def mat_mul(m1, m2, p):
        a1, b1, c1, d1 = m1
        a2, b2, c2, d2 = m2
        return ((a1*a2 + b1*c2) % p, (a1*b2 + b1*d2) % p,
                (c1*a2 + d1*c2) % p, (c1*b2 + d1*d2) % p)
    
    def mat_inv(m, p):
        a, b, c, d = m
        return (d % p, (-b) % p, (-c) % p, a % p)
    
    def canonical(elem):
        a, b, c, d = elem
        # In PSL(2,p), (a,b,c,d) ~ (-a,-b,-c,-d)
        # Canonical: make first non-zero component positive
        if a != 0:
            if a > p // 2:
                return ((-a) % p, (-b) % p, (-c) % p, (-d) % p)
            return elem
        elif b != 0:
            if b > p // 2:
                return ((-a) % p, (-b) % p, (-c) % p, (-d) % p)
            return elem
        elif c != 0:
            if c > p // 2:
                return ((-a) % p, (-b) % p, (-c) % p, (-d) % p)
            return elem
        else:
            if d > p // 2:
                return ((-a) % p, (-b) % p, (-c) % p, (-d) % p)
            return elem
    
    # Build directed voltage map
    dir_volt = [None] * (2 * n_edges)
    for i, (u, v) in enumerate(base_edges):
        m = voltages[(u, v)]
        dir_volt[i] = m
        dir_volt[i + n_edges] = mat_inv(m, p)
    
    # BFS from (0, identity)
    identity = (1, 0, 0, 1)
    start_state = (0, canonical(identity))
    
    visited = {}  # (base_v, elem) -> vertex_index
    order = []
    queue = deque([start_state])
    visited[start_state] = 0
    order.append(start_state)
    
    while queue:
        bv, elem = queue.popleft()
        for nv in base_adj[bv]:
            # Find directed edge index
            # Need to find which edge connects bv and nv
            # Precompute for efficiency
            # For now, search
            for i, (u, v) in enumerate(base_edges):
                if u == bv and v == nv:
                    e_idx = i
                    break
                elif u == nv and v == bv:
                    e_idx = i + n_edges
                    break
            else:
                continue
            
            new_elem = mat_mul(elem, dir_volt[e_idx], p)
            new_elem = canonical(new_elem)
            new_state = (nv, new_elem)
            
            if new_state not in visited:
                visited[new_state] = len(order)
                order.append(new_state)
                queue.append(new_state)
    
    # Build edge list
    edges = []
    for bv, elem in order:
        u_idx = visited[(bv, elem)]
        for nv in base_adj[bv]:
            for i, (u, v) in enumerate(base_edges):
                if u == bv and v == nv:
                    e_idx = i
                    break
                elif u == nv and v == bv:
                    e_idx = i + n_edges
                    break
            else:
                continue
            
            new_elem = mat_mul(elem, dir_volt[e_idx], p)
            new_elem = canonical(new_elem)
            v_idx = visited.get((nv, new_elem))
            if v_idx is not None and u_idx < v_idx:
                edges.append((u_idx, v_idx))
    
    # Verify 3-regular
    n_vertices = len(order)
    if len(edges) != 3 * n_vertices // 2:
        return None
    
    return edges

def _luw_fallback(k, g):
    """Fallback: Lazebnik-Ustimenko-Woldar construction."""
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
