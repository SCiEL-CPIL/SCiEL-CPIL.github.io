import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    def _deadline_check():
        return time.time() > deadline

    def verify_girth(adj, g):
        """Verify girth >= g using non-backtracking BFS from every vertex."""
        n = len(adj)
        if n == 0:
            return True
        radius = (g - 1) // 2
        for u in range(n):
            visited = {}
            visited[u] = (-1, 0)
            queue = deque([(u, -1, 0)])
            while queue:
                node, parent, dist = queue.popleft()
                if dist == radius:
                    continue
                for nei in adj[node]:
                    if nei == parent:
                        continue
                    if nei in visited:
                        return False
                    visited[nei] = (node, dist + 1)
                    queue.append((nei, node, dist + 1))
        return True

    def luw_construct(k, g):
        """LUW construction for 3-regular graphs."""
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

    # Get LUW baseline
    n_luw, edges_luw = luw_construct(k, g)
    best_n = n_luw
    best_edges = edges_luw

    # Moore bound for k=3, g=16: n >= 2 * (2^8 - 1) = 510
    moore_bound = 2 * ((k - 1) ** (g // 2) - 1) // (k - 2) if k > 2 else 1

    # Try voltage lifts of small base graphs
    # Strategy: use Petersen graph (10 vertices, 3-regular) as base
    # Lift by Z_m gives 10*m vertices
    # For girth 16, we need m large enough that random lifts avoid short cycles
    
    # Base graph: Petersen graph
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # outer 5-cycle
        (5, 6), (6, 7), (7, 8), (8, 9), (9, 5),  # inner 5-cycle
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)   # spokes
    ]
    petersen_adj = [[] for _ in range(10)]
    for u, v in petersen_edges:
        petersen_adj[u].append(v)
        petersen_adj[v].append(u)
    
    # Also try K4 (4 vertices) and other small 3-regular graphs
    # K4 is 3-regular with 4 vertices
    k4_edges = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    k4_adj = [[] for _ in range(4)]
    for u, v in k4_edges:
        k4_adj[u].append(v)
        k4_adj[v].append(u)
    
    # Try voltage lifts
    # For Petersen: n = 10*m, so m = n/10
    # For K4: n = 4*m, so m = n/4
    
    base_graphs = [
        (petersen_adj, petersen_edges, 10),  # (adj, edges, n_base)
        (k4_adj, k4_edges, 4),
    ]
    
    # Search for smaller n using voltage lifts
    # Target: get below LUW's n
    search_start = min(best_n - 10, 1000)
    search_end = max(moore_bound, 510)
    
    if search_start >= search_end:
        search_start = best_n - 10
    
    # Try various m values (voltage group sizes)
    # For Petersen: n = 10*m, so try m from search_end/10 to search_start/10
    # For K4: n = 4*m, so try m from search_end/4 to search_start/4
    
    time_budget_voltage = 50  # seconds for voltage lift search
    voltage_deadline = start_time + time_budget_voltage
    
    # Try Petersen lifts first (more promising for high girth)
    for n_base, base_adj, base_edges_list in [(10, petersen_adj, petersen_edges), (4, k4_adj, k4_edges)]:
        if time.time() > voltage_deadline:
            break
        
        # Try m values that give n between search_end and search_start
        m_min = max(1, (search_end + n_base - 1) // n_base)
        m_max = search_start // n_base
        
        if m_max < m_min:
            continue
        
        # Try a range of m values, prioritizing smaller n
        # For each m, try several random voltage assignments
        m_values = list(range(m_min, min(m_max + 1, m_min + 200)))
        random.shuffle(m_values)
        # Sort by resulting n (ascending) to try smaller n first
        m_values.sort(key=lambda m: n_base * m)
        
        for m in m_values:
            if time.time() > voltage_deadline:
                break
            
            n_lift = n_base * m
            if n_lift > search_start:
                continue
            if n_lift < search_end:
                continue
            
            # Try up to 5 random voltage assignments for this m
            max_trials = 5
            for trial in range(max_trials):
                if time.time() > voltage_deadline:
                    break
                
                # Random voltage assignment: assign random elements of Z_m to each edge
                voltages = [random.randint(0, m - 1) for _ in range(len(base_edges_list))]
                
                # Build the lift
                # Vertex (base_v, fiber_f) has neighbors (base_u, fiber_f + voltage(e)) for each edge (base_v, base_u)
                # We need to be careful about direction
                lift_adj = [[] for _ in range(n_lift)]
                valid = True
                
                for ei, (u, v) in enumerate(base_edges_list):
                    vol = voltages[ei]
                    for f in range(m):
                        # Edge from (u, f) to (v, f + vol mod m)
                        a = u * m + f
                        b = v * m + ((f + vol) % m)
                        lift_adj[a].append(b)
                        # Edge from (v, f - vol mod m) to (u, f)
                        # Actually, the edge (u,v) with voltage vol means:
                        # (u, f) connects to (v, f + vol)
                        # (v, f - vol) connects to (u, f)
                        # So we add both directions
                        lift_adj[b].append(a)
                
                # Check if simple (no multi-edges)
                for i in range(n_lift):
                    if len(lift_adj[i]) != 3:
                        valid = False
                        break
                    if len(set(lift_adj[i])) != 3:
                        valid = False
                        break
                    if i in lift_adj[i]:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Verify girth
                if verify_girth(lift_adj, g):
                    edges = []
                    for i in range(n_lift):
                        for j in lift_adj[i]:
                            if i < j:
                                edges.append((i, j))
                    return n_lift, edges

    # If voltage lifts didn't work, try random 3-regular graphs for smaller n
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        # Try random graphs for n values below LUW
        for n in range(min(search_start, 900), max(search_end, 510) - 1, -1):
            if time.time() > deadline - 5:
                break
            
            # Try a few random graphs
            max_trials = 3
            for trial in range(max_trials):
                if time.time() > deadline - 5:
                    break
                
                adj = _generate_random_3_regular(n)
                if adj is None:
                    continue
                if verify_girth(adj, g):
                    edges = []
                    for i in range(n):
                        for j in adj[i]:
                            if i < j:
                                edges.append((i, j))
                    return n, edges

    return best_n, best_edges

def _generate_random_3_regular(n):
    """Generate a random 3-regular graph using configuration model."""
    if n < 3 or (3 * n) % 2 != 0:
        return None
    
    # Configuration model
    half_edges = list(range(3 * n))
    random.shuffle(half_edges)
    
    adj = [[] for _ in range(n)]
    valid = True
    for i in range(0, 3 * n, 2):
        u = half_edges[i] // 3
        v = half_edges[i + 1] // 3
        
        if u == v:
            valid = False
            break
        
        if v in adj[u]:
            valid = False
            break
        
        adj[u].append(v)
        adj[v].append(u)
    
    if not valid:
        return None
    
    for i in range(n):
        if len(adj[i]) != 3:
            return None
        if len(set(adj[i])) != 3:
            return None
    
    return adj
