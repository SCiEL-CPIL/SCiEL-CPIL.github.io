import time
import random
from collections import deque

# EVOLVE-BLOCK-START

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Fallback: The seed LUW graph
    def luw_construct(k, g):
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

    # Get fallback result immediately to have a valid answer
    fallback_n, fallback_edges = luw_construct(k, g)
    best_n = fallback_n
    best_edges = fallback_edges
    
    # Fast girth check using BFS with early termination
    def check_girth_fast(adj, n, g):
        """Check if graph has girth >= g. Returns True if valid."""
        if n < g:
            return False
        max_depth = (g - 1) // 2
        
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                du = dist[u]
                if du >= max_depth:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = du + 1
                        parent[v] = u
                        if du + 1 < max_depth:
                            q.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
        return True

    # Build base graph and try voltage lifts
    def try_voltage_lift(base_edges, n_base, m, voltages, g):
        """Build lifted graph and check girth. Returns (n_lifted, edges) if valid, else None."""
        n_lifted = n_base * m
        adj = [[] for _ in range(n_lifted)]
        lifted_edges = []
        
        for u, v in base_edges:
            w = voltages.get((u, v), 0)
            for i in range(m):
                ui = u * m + i
                vj = v * m + (i + w) % m
                adj[ui].append(vj)
                adj[vj].append(ui)
                lifted_edges.append((ui, vj))
        
        # Verify 3-regularity
        for i in range(n_lifted):
            if len(adj[i]) != k:
                return None
        
        # Check girth
        if not check_girth_fast(adj, n_lifted, g):
            return None
        
        return n_lifted, lifted_edges

    # Define base graphs
    # 1. Heawood Graph (14 vertices, girth 6)
    fano_lines = [
        (0, 1, 2), (0, 3, 4), (0, 5, 6),
        (1, 3, 5), (1, 4, 6), (2, 3, 6), (2, 4, 5)
    ]
    heawood_edges = []
    for i, line in enumerate(fano_lines):
        line_id = 7 + i
        for p in line:
            heawood_edges.append((p, line_id))
    
    # 2. K3,3 (6 vertices, girth 4)
    k33_edges = [(0,3), (0,4), (0,5), (1,3), (1,4), (1,5), (2,3), (2,4), (2,5)]
    
    # 3. Petersen Graph (10 vertices, girth 5)
    petersen_edges = [(0,1), (1,2), (2,3), (3,4), (4,0),
                      (5,7), (7,9), (9,6), (6,8), (8,5),
                      (0,5), (1,7), (2,9), (3,6), (4,8)]
    
    base_graphs = [
        ("heawood", heawood_edges, 14),
        ("k33", k33_edges, 6),
        ("petersen", petersen_edges, 10),
    ]
    
    # Moore bound for k=3, g=16: n >= 2 * sum_{i=0}^{7} 2^i = 510
    moore_bound = 2 * sum(2**i for i in range(g // 2))
    
    # Try voltage lifts
    for base_name, base_edges, n_base in base_graphs:
        if time.time() > deadline:
            break
        
        # Minimum m such that n_base * m >= moore_bound
        min_m = max(1, (moore_bound + n_base - 1) // n_base)
        
        # Try m values
        m_candidates = list(range(min_m, min(min_m + 40, 300)))
        for extra in [300, 400, 500, 700, 900]:
            if extra > min_m and extra < best_n // n_base:
                m_candidates.append(extra)
        m_candidates = sorted(set(m_candidates))
        
        for m in m_candidates:
            if time.time() > deadline:
                break
            if n_base * m >= best_n:
                break
            
            # Try random voltage assignments
            num_trials = 15
            for trial in range(num_trials):
                if time.time() > deadline:
                    break
                
                voltages = {}
                for u, v in base_edges:
                    voltages[(u, v)] = random.randint(1, m - 1)
                
                result = try_voltage_lift(base_edges, n_base, m, voltages, g)
                if result is not None:
                    n_lifted, lifted_edges = result
                    if n_lifted < best_n:
                        best_n = n_lifted
                        best_edges = lifted_edges
            
            # Try a deterministic pattern
            voltages_det = {}
            for i, (u, v) in enumerate(base_edges):
                voltages_det[(u, v)] = (i + 1) % m if m > 1 else 0
            
            result = try_voltage_lift(base_edges, n_base, m, voltages_det, g)
            if result is not None:
                n_lifted, lifted_edges = result
                if n_lifted < best_n:
                    best_n = n_lifted
                    best_edges = lifted_edges
    
    return best_n, best_edges

# EVOLVE-BLOCK-END
