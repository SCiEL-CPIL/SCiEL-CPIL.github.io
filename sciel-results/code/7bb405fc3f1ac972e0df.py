import time
import random
from collections import deque

# EVOLVE-BLOCK-START

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k (bipartite)."""
    q = k
    # Choose d such that the girth is at least g
    # For LUW graphs, girth is 2*ceil(d/2)+1 or similar; use d >= g
    d = g
    if d % 2 == 0:
        d += 1
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

def _abs_construct(k, g):
    """
    Alon-Babai-Suzuki (ABS) type construction:
    Cayley graph on the group of upper unitriangular 3x3 matrices over F_q.
    Group elements are (a, b, c) with multiplication:
      (a,b,c) * (a',b',c') = (a+a', b+b', c+c' + a*b')
    This is the Heisenberg group over F_q.
    
    We use a symmetric generating set to get a k-regular graph.
    For k=5, we can use generators: (1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1)
    but (0,0,1) and its inverse (0,0,-1) are distinct, so that's 6 generators.
    
    Let's think more carefully. The group has q^3 elements.
    We need a symmetric generating set S with |S| = k, S = S^{-1}, 0 not in S.
    
    For k=5 (odd), we need a symmetric set of size 5. This means we need pairs {s, s^{-1}} 
    plus one element s with s = s^{-1} (i.e., s has order 2).
    
    In the Heisenberg group, elements of order 2: (a,b,c) has order 2 iff 2(a,b,c) = (0,0,0).
    2(a,b,c) = (2a, 2b, 2c + 2ab). So we need 2a=0, 2b=0, 2c+2ab=0.
    For odd q, 2a=0 implies a=0, so b=0, c=0. No non-trivial elements of order 2 for odd q.
    
    For q=5 (odd), we can't easily get a symmetric set of odd size.
    
    Alternative: use a different group or a different approach.
    
    Let's try the group of affine transformations or a different unitriangular group.
    
    Actually, let's use a voltage graph / Cayley graph on Z_q x Z_q with a specific generating set.
    
    Better approach: Use the construction from Alon, Babai, Suzuki (1992) for non-bipartite cages.
    They use Cayley graphs on finite groups. For girth g and degree k, they construct a graph 
    on approximately q^(g/2) vertices where q is a prime power.
    
    For our purposes, let's try a simpler approach: construct a Cayley graph on Z_q^m 
    with a carefully chosen generating set that avoids short cycles.
    
    For k=5, g=9:
    Try q=5, m=3, so n=125. But Moore bound is 428, so 125 < 428, impossible.
    Try q=7, m=3, n=343 < 428.
    Try q=11, m=3, n=1331.
    Try q=5, m=4, n=625.
    Try q=7, m=4, n=2401.
    
    Let's try q=5, m=4, n=625 with a generating set that gives high girth.
    """
    q = 5
    m = 4
    n = q ** m
    
    # Generating set for Z_q^m: we need k=5 symmetric generators
    # Use: e1, -e1, e2, -e2, e3 where e_i are standard basis vectors
    # But this gives bipartite-like structure. Let's use a different set.
    
    # For a non-bipartite graph, we need an odd cycle in the generator set.
    # Use generators: (1,0,0,0), (0,1,0,0), (0,0,1,0), (-1,-1,0,0), (0,0,0,1)
    # Check symmetry: inverse of (a,b,c,d) is (-a,-b,-c,-d)
    # (1,0,0,0)^{-1} = (-1,0,0,0) - not in set
    # This is getting complicated.
    
    # Let's just use the LUW construction as fallback and try voltage lifts more aggressively.
    return None, None

def _generate_random_regular(k, n, seed=None):
    """Generate a random k-regular graph on n vertices using configuration model."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random
    
    if n * k % 2 != 0:
        return None
        
    half_edges = []
    for v in range(n):
        half_edges.extend([v] * k)
    
    rng.shuffle(half_edges)
    
    edges = set()
    adj = {v: set() for v in range(n)}
    
    for i in range(0, n * k, 2):
        u, v = half_edges[i], half_edges[i + 1]
        if u == v:
            continue
        if u in adj[v]:
            continue
        edges.add((min(u, v), max(u, v)))
        adj[u].add(v)
        adj[v].add(u)
    
    for v in range(n):
        if len(adj[v]) != k:
            return None
    
    return list(edges)

def _construct_lift(base_edges, base_n, m, voltage_seed, k):
    """Construct the lift of a base graph with voltages in Z_m."""
    rng = random.Random(voltage_seed)
    
    voltages = {}
    for u, v in base_edges:
        voltages[(u, v)] = rng.randint(0, m - 1)
    
    n = base_n * m
    
    lift_edges = set()
    adj = {i: [] for i in range(n)}
    
    for u, v in base_edges:
        vol = voltages[(u, v)]
        for a in range(m):
            u_lift = u * m + a
            v_lift = v * m + (a + vol) % m
            edge = (min(u_lift, v_lift), max(u_lift, v_lift))
            if edge in lift_edges:
                return None, None
            lift_edges.add(edge)
            adj[u_lift].append(v_lift)
            adj[v_lift].append(u_lift)
    
    for i in range(n):
        if len(adj[i]) != len(set(adj[i])):
            return None, None
        if len(adj[i]) != k:
            return None, None
            
    return n, list(lift_edges)

def _has_short_cycle(adj, n, g, k):
    """Check if graph has a cycle of length < g using BFS from each vertex."""
    r = g // 2
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            if dist[u] >= r:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    if dist[v] < r:
                        q.append(v)
                elif v != parent[u] and u != parent[v]:
                    uu, vv = u, v
                    du, dv = dist[u], dist[v]
                    while du > dv:
                        uu = parent[uu]
                        du -= 1
                    while dv > du:
                        vv = parent[vv]
                        dv -= 1
                    while uu != vv:
                        uu = parent[uu]
                        vv = parent[vv]
                    lca_dist = du
                    cycle_len = dist[u] + dist[v] + 1 - 2 * lca_dist
                    if cycle_len < g:
                        return True
    return False

def _has_short_cycle_fast(adj, n, g):
    """Faster girth check: only BFS up to depth r-1, check for back edges."""
    r = (g - 1) // 2  # max depth to explore
    for s in range(n):
        dist = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            if dist[u] >= r:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    if dist[v] < r:
                        q.append(v)
                elif dist[v] <= dist[u]:
                    # Back edge or cross edge
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return True
    return False

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_n = float('inf')
    best_edges = None
    
    # Moore bound for (k,g)
    if g % 2 == 0:
        moore = 1 + k * sum(k**(i-1) for i in range(1, g//2))
    else:
        moore = 1 + k * sum(k**(i-1) for i in range(1, g//2)) + k * (k-1) ** (g//2 - 1)
    
    # Try voltage graph lifts with various base sizes and lift sizes
    base_sizes = [10, 12, 14, 16, 18, 20, 22, 24, 28, 30]
    
    for base_n in base_sizes:
        if time.time() > deadline:
            break
        
        num_attempts = 5
        for attempt in range(num_attempts):
            if time.time() > deadline:
                break
            
            seed = random.randint(0, 2**31)
            base_edges = _generate_random_regular(k, base_n, seed)
            if base_edges is None:
                continue
            
            # Try various lift sizes m
            for m in range(3, 100):
                if time.time() > deadline:
                    break
                n_candidate = base_n * m
                if n_candidate < moore:
                    continue
                if n_candidate > 3000:
                    break
                
                # Try a few voltage assignments
                for vs in range(2):
                    if time.time() > deadline:
                        break
                    
                    n, edges = _construct_lift(base_edges, base_n, m, seed + vs, k)
                    if n is None:
                        continue
                    
                    # Check girth
                    adj = {i: [] for i in range(n)}
                    for u, v in edges:
                        adj[u].append(v)
                        adj[v].append(u)
                    
                    if not _has_short_cycle_fast(adj, n, g):
                        if n < best_n:
                            best_n = n
                            best_edges = edges
    
    if best_edges is not None:
        return best_n, best_edges
    
    # Fallback to LUW construction
    n, edges = _luw_construct(k, g)
    return n, edges

# EVOLVE-BLOCK-END
