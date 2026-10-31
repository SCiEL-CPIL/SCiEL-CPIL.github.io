"""Construct a 3-regular graph with girth >= g on as few vertices as possible."""
import time
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    """
    Construct a k-regular graph with girth >= g using a voltage-lift construction.
    
    For k=3, g=16: Use a deterministic voltage lift of the 3-regular base graph
    (the Petersen graph or similar) over Z_m with carefully chosen voltages to
    maximize girth while minimizing total vertices.
    
    Strategy: 
    1. Start with a small 3-regular base graph B with n_b vertices.
    2. Assign voltages (elements of Z_m) to each directed edge.
    3. The lift has n_b * m vertices.
    4. Search over m and voltage assignments to find smallest graph with girth >= g.
    
    For girth 16, we need to ensure no closed walk of length < 16 in the lift.
    A closed walk in the lift projects to a closed walk in B with total voltage = 0 mod m.
    """
    deadline = time.time() + 145
    
    def check_girth_bfs(n, adj, k, g):
        """Check if graph has girth >= g using BFS from each vertex."""
        radius = g // 2 - 1
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            count = [0] * (radius + 1)
            count[0] = 1
            while queue:
                u = queue.popleft()
                if dist[u] >= radius:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] <= radius:
                            count[dist[v]] += 1
                            queue.append(v)
            expected = [1] + [k * (k-1)**i for i in range(radius)]
            if count != expected:
                return False
        return True
    
    # Base graph: Petersen graph (10 vertices, 3-regular, girth 5)
    # Petersen graph edges
    petersen_edges = [
        (0,1), (1,2), (2,3), (3,4), (4,0),  # outer 5-cycle
        (5,6), (6,7), (7,8), (8,9), (9,5),  # inner 5-cycle (pentagram)
        (0,5), (1,6), (2,7), (3,8), (4,9)   # spokes
    ]
    n_base = 10
    base_adj = [[] for _ in range(n_base)]
    for u, v in petersen_edges:
        base_adj[u].append(v)
        base_adj[v].append(u)
    
    def build_lift(m, voltages):
        """
        Build the voltage lift.
        voltages: dict mapping (u, v) -> voltage in Z_m for each directed edge
        """
        n = n_base * m
        adj = [[] for _ in range(n)]
        
        # Vertex (i, j) in base lifts to vertices (i, j + t) for t in Z_m
        # Edge (i, j) -> (i', j') in base with voltage w lifts to:
        #   (i, a) -> (i', a + w) for all a in Z_m
        
        for u in range(n_base):
            for v in base_adj[u]:
                w = voltages.get((u, v), 0)
                for a in range(m):
                    src = u * m + a
                    dst = v * m + ((a + w) % m)
                    adj[src].append(dst)
        
        # Remove duplicates and self-loops, make undirected
        final_adj = [[] for _ in range(n)]
        for v in range(n):
            seen = set()
            for u in adj[v]:
                if u != v and u not in seen:
                    final_adj[v].append(u)
                    seen.add(u)
                    if v not in final_adj[u]:
                        final_adj[u].append(v)
        
        # Check 3-regular
        for v in range(n):
            if len(final_adj[v]) != 3:
                return None
        
        return final_adj
    
    def generate_voltages(m, seed_pattern):
        """Generate a voltage assignment for the base graph edges."""
        voltages = {}
        # Use a structured pattern based on seed_pattern
        for u in range(n_base):
            for v in base_adj[u]:
                if u < v:
                    # Deterministic voltage based on edge position and seed
                    idx = petersen_edges.index((u, v))
                    voltages[(u, v)] = (seed_pattern * (idx + 1) + u * 3 + v) % m
                    voltages[(v, u)] = (-voltages[(u, v)]) % m
        return voltages
    
    best_result = None
    best_n = float('inf')
    
    # Search over m values
    # Moore bound for k=3, g=16 is 378, known lower bound 512.
    # Lift has 10*m vertices, so m >= 52 for n >= 520.
    # Try m from 52 upward, but also try smaller m to see if girth is satisfied.
    
    m_start = 52
    m_end = 80
    
    for m in range(m_start, m_end + 1):
        if time.time() > deadline:
            break
        
        # Try a few voltage patterns
        for sp in range(1, 6):
            if time.time() > deadline:
                break
            
            voltages = generate_voltages(m, sp)
            adj = build_lift(m, voltages)
            if adj is None:
                continue
            
            n = len(adj)
            if n >= best_n:
                continue
            
            # Quick connectivity check
            visited = [False] * n
            queue = deque([0])
            visited[0] = True
            count = 1
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        count += 1
                        queue.append(v)
            if count != n:
                continue
            
            # Check girth
            if check_girth_bfs(n, adj, k, g):
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                best_result = (n, edges)
                best_n = n
    
    if best_result is not None:
        return best_result
    
    # Fallback: LUW construction
    q = k
    d = g - 5 if (g - 5) % 2 else g - 4
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
# EVOLVE-BLOCK-END
