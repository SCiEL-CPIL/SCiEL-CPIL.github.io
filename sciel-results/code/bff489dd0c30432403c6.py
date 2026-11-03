import time
import random

# EVOLVE-BLOCK-START

def _girth_check(n, adj, g_target):
    """Check if girth is at least g_target using non-backtracking BFS from every vertex."""
    # For girth >= g_target, we need to verify no cycle of length < g_target.
    # A cycle of length L < g_target means there exists a non-backtracking walk
    # of length L from some vertex back to itself.
    # We check from each vertex, doing BFS up to depth floor((g_target-1)/2)
    # and checking if we can return to start.
    
    max_depth = (g_target - 1) // 2  # For g=10, max_depth=4
    
    for s in range(n):
        # BFS from s, tracking (current, parent, depth)
        # We want to detect if we can reach s again via a non-backtracking path
        # of length < g_target
        visited = {}  # (node, parent) -> depth
        queue = []
        for nb in adj[s]:
            state = (nb, s)
            visited[state] = 1
            queue.append((nb, s, 1))
        
        head = 0
        while head < len(queue):
            v, p, depth = queue[head]
            head += 1
            
            if depth >= max_depth:
                continue
            
            for nb in adj[v]:
                if nb == p:
                    continue
                if nb == s:
                    # Found a cycle of length depth + 1
                    if depth + 1 < g_target:
                        return False
                    continue
                state = (nb, v)
                if state not in visited:
                    visited[state] = depth + 1
                    queue.append((nb, v, depth + 1))
    
    return True

def _build_lift(base_edges, voltages, M):
    """Build the voltage graph lift.
    base_edges: list of (u, v) where u < v in base graph
    voltages: dict mapping (u, v) -> voltage for edge u->v
    M: size of voltage group Z_M
    """
    # Get all base vertices
    base_vertices = set()
    for u, v in base_edges:
        base_vertices.add(u)
        base_vertices.add(v)
    base_vertices = sorted(base_vertices)
    base_to_idx = {v: i for i, v in enumerate(base_vertices)}
    n_base = len(base_vertices)
    n = M * n_base
    
    edges = []
    for u, v in base_edges:
        w = voltages.get((u, v), 0)
        for x in range(M):
            u_id = base_to_idx[u] * M + x
            v_id = base_to_idx[v] * M + (x + w) % M
            edges.append((u_id, v_id))
    
    return n, edges

def construct(k, g):
    deadline = time.time() + 140
    
    # Specific case for (4, 10) cage
    if k == 4 and g == 10:
        # Use bipartite double cover of K5 as base graph
        # Base vertices: 0..4 (top), 5..9 (bottom)
        # Edges: (i, 5+j) for i != j
        base_edges = []
        for i in range(5):
            for j in range(5):
                if i != j:
                    base_edges.append((i, 5 + j))
        
        M = 32
        n_expected = M * 10  # 320 vertices
        
        # Try random voltage assignments
        for attempt in range(5000):
            if time.time() > deadline - 15:
                break
            
            # Random voltages in Z_32
            voltages = {}
            for u, v in base_edges:
                voltages[(u, v)] = random.randint(0, M - 1)
            
            n, edges = _build_lift(base_edges, voltages, M)
            
            # Build adjacency list for girth check
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            if _girth_check(n, adj, g):
                return n, edges
        
        # Fallback: Try K4,4 base with M=40
        base_edges_k44 = []
        for i in range(4):
            for j in range(4):
                base_edges_k44.append((i, j + 4))
        
        M2 = 40
        for attempt in range(2000):
            if time.time() > deadline - 10:
                break
            
            voltages = {}
            for u, v in base_edges_k44:
                voltages[(u, v)] = random.randint(0, M2 - 1)
            
            n, edges = _build_lift(base_edges_k44, voltages, M2)
            
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            if _girth_check(n, adj, g):
                return n, edges
    
    # General fallback: LUW construction
    return _luw(k, g)

def _luw(k, g):
    """Lazebnik-Ustimenko-Woldar graph construction."""
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

# EVOLVE-BLOCK-END
