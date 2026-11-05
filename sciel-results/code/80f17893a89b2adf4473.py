import time
import random
from collections import deque

# EVOLVE-BLOCK-START

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

def _check_girth(adj, n, g):
    """Check if graph has girth >= g using BFS with parent tracking."""
    r = (g - 1) // 2
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            du = dist[u]
            if du >= r:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    parent[v] = u
                    if dist[v] < r:
                        q.append(v)
                elif v != parent[u] and parent[v] != u:
                    uu, vv = u, v
                    du2, dv2 = du, dist[v]
                    while du2 > dv2:
                        uu = parent[uu]
                        du2 -= 1
                    while dv2 > du2:
                        vv = parent[vv]
                        dv2 -= 1
                    while uu != vv:
                        uu = parent[uu]
                        vv = parent[vv]
                        du2 -= 1
                        dv2 -= 1
                    lca_dist = du2
                    cycle_len = du + dist[v] + 1 - 2 * lca_dist
                    if cycle_len < g:
                        return False
    return True

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

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    best_n = float('inf')
    best_edges = None
    
    # Fallback: LUW construction (always valid, but large)
    n_luw, edges_luw = _luw_construct(k, g)
    best_n = n_luw
    best_edges = edges_luw
    
    # Strategy: Voltage graph lifts to get smaller graphs
    # For k=5, g=9: need 5-regular graph with girth >= 9
    # Moore bound: 2*(4^4) = 512 for odd girth 9
    # So we're looking for n in [512, 15625]
    
    # Try voltage lifts of small base graphs
    # Base must be k-regular (5-regular)
    # n_lift = base_n * m
    # We want n_lift < best_n
    
    # Try various base sizes
    base_sizes = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    
    for base_n in base_sizes:
        if time.time() > deadline:
            break
        
        # Generate base graph
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            seed = base_n * 10000 + attempt * 1000
            base_edges = _generate_random_regular(k, base_n, seed=seed)
            if base_edges is None:
                continue
            
            # Try different lift factors
            max_m = min(200, (best_n - 1) // base_n) if base_n > 0 else 0
            for m in range(2, max_m + 1):
                if time.time() > deadline:
                    break
                
                n_candidate = base_n * m
                if n_candidate >= best_n:
                    break
                
                # Try a few voltage assignments
                for vs in range(3):
                    if time.time() > deadline:
                        break
                    
                    vseed = seed + m * 10 + vs
                    n, edges = _construct_lift(base_edges, base_n, m, vseed, k)
                    if n is None:
                        continue
                    
                    # Check girth
                    adj = {i: [] for i in range(n)}
                    for u, v in edges:
                        adj[u].append(v)
                        adj[v].append(u)
                    
                    if _check_girth(adj, n, g):
                        if n < best_n:
                            best_n = n
                            best_edges = edges
    
    return best_n, best_edges

# EVOLVE-BLOCK-END
