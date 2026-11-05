import time
import random
import itertools
from collections import deque

def _get_icosahedron():
    """
    Returns the edge list and adjacency list of the Icosahedron graph (12 vertices, 5-regular, girth 5).
    """
    adj = [
        [1, 2, 3, 4, 5],
        [0, 2, 6, 7, 8],
        [0, 1, 6, 9, 10],
        [0, 4, 5, 9, 11],
        [0, 3, 5, 7, 11],
        [0, 3, 4, 8, 10],
        [1, 2, 7, 9, 11],
        [1, 4, 6, 8, 10],
        [1, 5, 7, 10, 11],
        [2, 3, 6, 8, 10],
        [2, 5, 7, 8, 9],
        [3, 4, 6, 8, 9]
    ]
    edges = []
    edge_map = {} # (u,v) -> index
    for u in range(12):
        for v in adj[u]:
            if u < v:
                idx = len(edges)
                edges.append((u, v))
                edge_map[(u, v)] = idx
                edge_map[(v, u)] = idx
    return edges, adj, edge_map

def _find_short_cycles(adj, edge_map, max_len):
    """
    Find all simple cycles of length < max_len in the base graph.
    Returns a list of lists of edge indices.
    """
    n = len(adj)
    cycles = []
    
    # DFS to find cycles
    # To avoid duplicates, we can enforce that the smallest vertex in the cycle is the start,
    # and the second smallest is the first step? 
    # Simpler: Just find all cycles and deduplicate by sorted edge set.
    cycle_sets = set()
    
    def dfs(start, current, visited, path_edges):
        if len(path_edges) >= max_len:
            return
        for neighbor in adj[current]:
            if neighbor == start and len(path_edges) >= 3:
                # Found a cycle
                # Normalize the cycle representation
                # The path_edges are indices. 
                # We need to ensure we don't add the same cycle multiple times.
                # A cycle is defined by the set of edges.
                edge_tuple = tuple(sorted(path_edges))
                if edge_tuple not in cycle_sets:
                    cycle_sets.add(edge_tuple)
                    cycles.append(list(path_edges))
            elif neighbor not in visited and neighbor > start: 
                # Restricting neighbor > start helps reduce duplicates significantly
                # and ensures start is the minimum vertex in the cycle
                visited.add(neighbor)
                path_edges.append(edge_map[(current, neighbor)])
                dfs(start, neighbor, visited, path_edges)
                path_edges.pop()
                visited.remove(neighbor)

    for s in range(n):
        visited = {s}
        dfs(s, s, visited, [])
        
    return cycles

def _solve_voltages(cycles, m, num_trials=200, max_steps=5000):
    """
    Attempt to find voltage assignments for edges (0..E-1) in Z_m
    such that no cycle in 'cycles' has a voltage sum of 0 mod m.
    Uses randomized local search.
    """
    E = len(cycles[0][0]) if cycles and cycles[0] else 0
    if not cycles:
        return None
    
    # Determine number of edges from the first cycle? No, we need total edges.
    # The 'cycles' list contains edge indices. The max index + 1 is E.
    if not cycles:
        E = 0
    else:
        max_idx = -1
        for c in cycles:
            for e in c:
                if e > max_idx:
                    max_idx = e
        E = max_idx + 1

    for _ in range(num_trials):
        # Initialize random voltages
        voltages = [random.randint(0, m - 1) for _ in range(E)]
        
        # Pre-compute cycle sums? Or just check on the fly.
        # For local search, we need to know which cycles are violated.
        
        for step in range(max_steps):
            violated_cycles = []
            for c in cycles:
                s = 0
                for e in c:
                    s = (s + voltages[e]) % m
                if s == 0:
                    violated_cycles.append(c)
                    # Early exit if we find one to fix
                    break
            
            if not violated_cycles:
                return voltages
            
            # Pick a random violated cycle
            c = random.choice(violated_cycles)
            # Pick a random edge in the cycle to modify
            e_idx = random.choice(c)
            # Change voltage to a random value
            voltages[e_idx] = random.randint(0, m - 1)
            
    return None

def _construct_lift(base_edges, m, voltages):
    """
    Construct the lift of base_edges with given voltages in Z_m.
    Returns (n, edges, adj) or None if invalid.
    """
    base_vertices = set()
    for u, v in base_edges:
        base_vertices.add(u)
        base_vertices.add(v)
    base_n = len(base_vertices)
    n = base_n * m
    adj = {i: [] for i in range(n)}
    edges_set = set()
    
    for i, (u, v) in enumerate(base_edges):
        vol = voltages[i] % m
        for a in range(m):
            u_lift = u * m + a
            v_lift = v * m + (a + vol) % m
            
            if u_lift == v_lift:
                return None, None, None
            if v_lift in adj[u_lift]:
                return None, None, None
                
            adj[u_lift].append(v_lift)
            adj[v_lift].append(u_lift)
            edges_set.add((min(u_lift, v_lift), max(u_lift, v_lift)))
            
    for i in range(n):
        if len(adj[i]) != 5:
            return None, None, None
            
    return n, list(edges_set), adj

def _check_girth_bfs(adj, n, target_girth):
    """
    Check if graph has girth at least target_girth.
    """
    for s in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[s] = 0
        parent[s] = -2
        queue = deque([s])
        
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif parent[u] != v and parent[v] != u:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < target_girth:
                        return False
    return True

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

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # 5s margin
    
    # Moore bound
    r = g // 2
    if g % 2 == 1:
        n_min = 1
        term = k
        for _ in range(r - 1):
            n_min += term
            term *= (k - 1)
    else:
        n_min = 0
        term = 1
        for _ in range(r):
            n_min += term
            term *= (k - 1)
        n_min *= 2
        
    # Use Icosahedron base
    base_edges, base_adj, base_edge_map = _get_icosahedron()
    base_n = 12
    base_cycles = _find_short_cycles(base_adj, base_edge_map, g)
    
    best_n = float('inf')
    best_edges = None
    
    # Start m from ceil(n_min / base_n)
    m_start = max(1, (n_min + base_n - 1) // base_n)
    
    # We can try m up to a reasonable limit, e.g., 100
    for m in range(m_start, 100):
        if time.time() > deadline:
            break
            
        n = base_n * m
        
        # Try to solve voltages
        voltages = _solve_voltages(base_cycles, m, num_trials=50, max_steps=2000)
        
        if voltages is not None:
            res_n, res_edges, adj = _construct_lift(base_edges, m, voltages)
            if res_n is not None:
                # Verify with BFS
                if _check_girth_bfs(adj, res_n, g):
                    best_n = n
                    best_edges = res_edges
                    break # Found smallest n for this base
            
    if best_n != float('inf'):
        return best_n, best_edges
        
    # Fallback
    n, edges = _luw_construct(k, g)
    return n, edges
