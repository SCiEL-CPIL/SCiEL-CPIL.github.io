"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
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


def _check_girth_nonbacktracking(adj, g, deadline=None):
    """
    Check if graph has girth at least g using non-backtracking walks.
    A cycle of length L < g would appear as a non-backtracking walk of length L
    that returns to the start vertex (with the constraint that we don't immediately
    backtrack).
    
    For efficiency, we use BFS on the state (vertex, prev_vertex) to find the
    shortest non-backtracking cycle.
    """
    n = len(adj)
    if n == 0:
        return True
    
    # State: (current_vertex, previous_vertex)
    # We want to find if there's a non-backtracking walk from (v, u) back to (u, v)
    # or more precisely, a cycle of length < g.
    
    # A more efficient method: for each directed edge (u, v), do a BFS of non-backtracking
    # walks starting from (u, v). If we can reach (v, u) in less than g-1 steps, 
    # we have a cycle of length <= g.
    # Actually, if we start at (u,v) and reach (v,u) in L steps, the cycle length is L+1.
    # So we need L+1 < g => L < g-1.
    
    # However, doing this for every edge is expensive.
    # Alternative: Just do a BFS from every vertex, tracking the parent, and check for
    # back edges that create short cycles. This is what the seed code tried but was buggy.
    
    # Let's use the standard algorithm:
    # For each vertex s, do a BFS. Maintain dist[s][v] and parent[s][v].
    # When we see an edge (u, v) such that parent[s][u] != v and parent[s][v] != u,
    # the cycle length is dist[s][u] + dist[s][v] + 1.
    
    # This is O(n * (n + m)). For n~1000, m~2500, this is 1000 * 3500 = 3.5M ops, which is fine.
    
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        queue = deque([s])
        
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif parent[u] != v and parent[v] != u:
                    # Found a cycle
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True


def _generate_random_regular(k, n, seed=None):
    """Generate a random k-regular graph on n vertices using configuration model."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random
    
    # Configuration model: create n*k half-edges, pair them randomly
    half_edges = []
    for v in range(n):
        half_edges.extend([v] * k)
    
    rng.shuffle(half_edges)
    
    edges = set()
    adj = {v: set() for v in range(n)}
    
    for i in range(0, n * k, 2):
        u, v = half_edges[i], half_edges[i + 1]
        if u == v:
            continue  # skip loops
        if u in adj[v]:
            continue  # skip multi-edges
        edges.add((min(u, v), max(u, v)))
        adj[u].add(v)
        adj[v].add(u)
    
    # Check if we got a valid k-regular graph
    for v in range(n):
        if len(adj[v]) != k:
            return None
    
    return list(edges)


def _construct_lift(base_edges, m, voltage_seed):
    """Construct the lift of a base graph with voltages in Z_m."""
    rng = random.Random(voltage_seed)
    
    # Get base vertices
    base_vertices = set()
    for u, v in base_edges:
        base_vertices.add(u)
        base_vertices.add(v)
    base_vertices = sorted(base_vertices)
    base_n = len(base_vertices)
    base_idx = {v: i for i, v in enumerate(base_vertices)}
    
    # Assign voltages: each edge gets a random voltage in Z_m
    voltages = {}
    for i, (u, v) in enumerate(base_edges):
        voltages[(u, v)] = rng.randint(0, m - 1)
    
    # Construct lift: n = base_n * m vertices
    n = base_n * m
    
    # Lift vertex (v, a) maps to v * m + a
    lift_edges = set()
    adj = {i: [] for i in range(n)}
    
    for u, v in base_edges:
        vol = voltages[(u, v)]
        for a in range(m):
            u_lift = base_idx[u] * m + a
            v_lift = base_idx[v] * m + (a + vol) % m
            lift_edges.add((min(u_lift, v_lift), max(u_lift, v_lift)))
            adj[u_lift].append(v_lift)
            adj[v_lift].append(u_lift)
    
    # Check degree
    for i in range(n):
        if len(adj[i]) != len(set(adj[i])):
            return None, None  # multi-edges exist
        if len(adj[i]) < 5:
            return None, None
    
    return n, list(lift_edges)


def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    # First try voltage graph lifts with small base graphs
    best_n = float('inf')
    best_edges = None
    
    base_sizes = [10, 12, 14, 16, 18, 20]
    
    for base_n in base_sizes:
        if time.time() > deadline:
            break
        
        # Generate several random base graphs
        num_attempts = 20
        for attempt in range(num_attempts):
            if time.time() > deadline:
                break
            
            seed = random.randint(0, 2**31)
            base_edges = _generate_random_regular(k, base_n, seed)
            if base_edges is None:
                continue
            
            # Try different lift sizes
            m_values = []
            for target_n in [500, 600, 700, 800, 900, 1000, 1100]:
                m = target_n // base_n
                if m >= 2:
                    m_values.append(m)
            m_values = list(set(m_values))
            m_values.sort()
            
            for m in m_values:
                if time.time() > deadline:
                    break
                
                # Try several voltage assignments
                for vs in range(10):
                    if time.time() > deadline:
                        break
                    
                    n, edges = _construct_lift(base_edges, m, seed + vs)
                    if n is None:
                        continue
                    
                    # Build adjacency list for girth check
                    adj = {i: [] for i in range(n)}
                    for u, v in edges:
                        adj[u].append(v)
                        adj[v].append(u)
                    
                    # Check girth
                    if _check_girth_nonbacktracking(adj, g, deadline):
                        if n < best_n:
                            best_n = n
                            best_edges = edges
    
    # If we found a valid lift, return it
    if best_edges is not None:
        return best_n, best_edges
    
    # Fallback to LUW construction
    n, edges = _luw_construct(k, g)
    return n, edges

# EVOLVE-BLOCK-END
