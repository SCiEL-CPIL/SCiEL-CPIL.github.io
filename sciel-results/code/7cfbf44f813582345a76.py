import time
import random
from collections import deque

# EVOLVE-BLOCK-START

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

def _construct_lift(base_edges, base_n, m, voltage_seed):
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
    """Check if graph has a cycle of length < g."""
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
                    # Potential cycle
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

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k."""
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

def _tensor_product_graph(graph1, graph2):
    """
    Construct the tensor (Kronecker) product of two regular graphs.
    If G1 is k1-regular on n1 vertices and G2 is k2-regular on n2 vertices,
    the tensor product G1 × G2 is (k1*k2)-regular on n1*n2 vertices.
    
    Vertices are pairs (u, v) where u ∈ V(G1), v ∈ V(G2).
    Two vertices (u1, v1) and (u2, v2) are adjacent iff:
      u1 ~ u2 in G1 AND v1 ~ v2 in G2
    
    Girth property: girth(G1 × G2) >= girth(G1) + girth(G2) - 1
    (More precisely, any cycle in the product projects to closed walks in both
    factors, and the cycle length is at least the sum of the shortest non-trivial
    closed walks that synchronize, which gives us a lower bound.)
    
    Actually, the girth of the tensor product satisfies:
    girth(G1 × G2) >= min(girth(G1), girth(G2)) * 2 in general, but for our purposes
    we need g1 + g2 >= g to ensure girth >= g.
    """
    n1, edges1 = graph1
    n2, edges2 = graph2
    
    # Build adjacency sets for both graphs
    adj1 = {i: set() for i in range(n1)}
    for u, v in edges1:
        adj1[u].add(v)
        adj1[v].add(u)
    
    adj2 = {i: set() for i in range(n2)}
    for u, v in edges2:
        adj2[u].add(v)
        adj2[v].add(u)
    
    # Verify regularity
    k1 = len(adj1[0])
    k2 = len(adj2[0])
    for v in range(n1):
        if len(adj1[v]) != k1:
            return None, None
    for v in range(n2):
        if len(adj2[v]) != k2:
            return None, None
    
    n = n1 * n2
    edges = []
    
    # For each edge in G1 and each edge in G2, create corresponding edges in product
    for u1, v1 in edges1:
        for u2, v2 in edges2:
            a = u1 * n2 + u2
            b = v1 * n2 + v2
            c = u1 * n2 + v2
            d = v1 * n2 + u2
            # (u1,u2) ~ (v1,v2) and (u1,v2) ~ (v1,u2)
            edges.append((min(a, b), max(a, b)))
            edges.append((min(c, d), max(c, d)))
    
    # Remove duplicates
    edges = list(set(edges))
    
    return n, edges

def _find_small_regular_with_girth(k, g, deadline, max_search_n=30):
    """Find a small k-regular graph with girth >= g."""
    best = None
    for n in range(k + 1, max_search_n + 1):
        if n * k % 2 != 0:
            continue
        # Moore bound check: n >= 2 * (k-1)^(floor((g-1)/2)) for odd girth
        # For g=9, floor(8/2)=4, so n >= 2*(k-1)^4
        # But we're looking for small graphs, so just try
        for attempt in range(20):
            if time.time() > deadline:
                break
            seed = random.randint(0, 2**31)
            edges = _generate_random_regular(k, n, seed)
            if edges is None:
                continue
            adj = {i: [] for i in range(n)}
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            if not _has_short_cycle(adj, n, g, k):
                if best is None or n < best[0]:
                    best = (n, edges)
                    if n <= k + 1:
                        return best
                break  # Found one for this n, try smaller
    return best

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_n = float('inf')
    best_edges = None
    
    # Strategy 1: Tensor product construction
    # We need k = k1 * k2, and girth(G1) + girth(G2) >= g
    # For k=5 (prime), we can only use k1=1, k2=5 or k1=5, k2=1
    # So tensor product doesn't directly work for prime k.
    # However, we can use a different approach:
    # Use the Cartesian product or strong product with voltage assignments.
    
    # Actually, for k=5, let's try:
    # 1. Voltage lifts (existing approach)
    # 2. LUW construction as fallback
    
    # Try voltage graph lifts
    base_sizes = [10, 12, 14, 16, 18, 20, 24, 30]
    
    for base_n in base_sizes:
        if time.time() > deadline:
            break
        
        num_attempts = 10
        for attempt in range(num_attempts):
            if time.time() > deadline:
                break
            
            seed = random.randint(0, 2**31)
            base_edges = _generate_random_regular(k, base_n, seed)
            if base_edges is None:
                continue
            
            # Try lift sizes m such that n = base_n * m is in a good range
            m_values = set()
            for target_n in range(500, 1500, 10):
                if target_n % base_n == 0:
                    m = target_n // base_n
                    if m >= 2:
                        m_values.add(m)
            
            # Also try specific small m
            for m in range(2, 50):
                if time.time() > deadline:
                    break
                n_candidate = base_n * m
                if n_candidate < 400:
                    continue
                if n_candidate > 2000:
                    continue
                
                # Try a few voltage assignments
                for vs in range(3):
                    if time.time() > deadline:
                        break
                    
                    n, edges = _construct_lift(base_edges, base_n, m, seed + vs)
                    if n is None:
                        continue
                    
                    # Check girth
                    adj = {i: [] for i in range(n)}
                    for u, v in edges:
                        adj[u].append(v)
                        adj[v].append(u)
                    
                    if not _has_short_cycle(adj, n, g, k):
                        if n < best_n:
                            best_n = n
                            best_edges = edges
    
    if best_edges is not None:
        return best_n, best_edges
    
    # Fallback
    n, edges = _luw_construct(k, g)
    return n, edges

# EVOLVE-BLOCK-END
