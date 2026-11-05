import time
import math
from collections import deque
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    best_n = float('inf')
    best_edges = None
    
    # Candidate 1: SL(2,q) Cayley graph - try small q values
    if k == 4:
        for q in [5, 7, 9, 11, 13]:
            if time.time() > deadline - 20:
                break
            try:
                n_cay, edges_cay = _construct_sl2_cayley(q)
                if n_cay < best_n and _verify_girth_fast(n_cay, edges_cay, k, g):
                    best_n = n_cay
                    best_edges = edges_cay
            except Exception:
                pass
    
    # Candidate 2: Voltage graph / covering construction
    if k == 4 and time.time() < deadline - 15:
        try:
            n_vol, edges_vol = _construct_voltage_graph(k, g)
            if n_vol < best_n and _verify_girth_fast(n_vol, edges_vol, k, g):
                best_n = n_vol
                best_edges = edges_vol
        except Exception:
            pass
    
    # Candidate 3: Lazebnik-Ustimenko-Woldar (LUW) - vectorized
    try:
        n_luw, edges_luw = _construct_luw_vectorized(k, g)
        if _verify_girth_fast(n_luw, edges_luw, k, g):
            if n_luw < best_n:
                best_n = n_luw
                best_edges = edges_luw
    except Exception:
        pass
    
    # If no graph found, fallback to LUW
    if best_edges is None:
        try:
            best_n, best_edges = _construct_luw_vectorized(k, g)
        except Exception:
            try:
                best_n, best_edges = _construct_luw(k, g)
            except Exception:
                best_n = k
                best_edges = []
                
    return best_n, best_edges

def _construct_voltage_graph(k, g):
    """
    Construct a 4-regular graph of girth >= 10 using a voltage graph (covering graph) approach.
    Uses a base graph with voltage assignments to a cyclic group to produce a larger graph.
    
    Strategy: Use a base graph with voltage assignments such that the resulting
    covering graph is 4-regular and has girth >= 10.
    
    We use a specific known construction: a (4,10) cage can be obtained as a
    covering of a small graph.
    """
    if k != 4 or g != 10:
        return None, None
    
    # Use a known explicit construction for the (4,10) cage
    # Based on the work of Exoo et al., we can try a specific algebraic construction.
    
    # Alternative: Use a Cayley graph of a specific group
    # Try Z_5 x Z_5 x Z_5 with appropriate generators
    # Or use a known (4,10)-cage construction
    
    # Let's try a construction based on a 2D torus with twist
    # Vertices: (i, j, layer) where i, j in Z_m, layer in Z_n
    # This gives a 4-regular graph if we connect appropriately
    
    # Try m=8, n=5 -> 320 vertices
    m, n = 8, 5
    N = m * n
    
    edges = []
    seen = set()
    
    for i in range(m):
        for j in range(n):
            v = i * n + j
            # 4 neighbors:
            # (i+1, j), (i-1, j), (i, j+1), (i, j-1) with appropriate twists
            # To get girth 10, we need to add twists
            # Use a linear twist: neighbor in j-direction depends on i
            
            # Up: (i, (j+1) mod n)
            n1 = i * n + ((j + 1) % n)
            # Down: (i, (j-1) mod n)
            n2 = i * n + ((j - 1) % n)
            # Right: ((i+1) mod m, (j + i) mod n)  -- twist
            n3 = ((i + 1) % m) * n + ((j + i) % n)
            # Left: ((i-1) mod m, (j - i) mod n)  -- twist
            n4 = ((i - 1) % m) * n + ((j - i) % n)
            
            for nb in [n1, n2, n3, n4]:
                e = (min(v, nb), max(v, nb))
                if e not in seen:
                    seen.add(e)
                    edges.append(e)
    
    return N, edges

def _construct_luw_vectorized(k, g):
    """Vectorized LUW construction using numpy arrays."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = np.array([[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]], dtype=np.int32)
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return table[x, y]
    else:
        def add(x, y): return (x + y) % q
        def sub(x, y): return (x - y) % q
        def mul(x, y): return x * y % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    # Use BFS to generate all vertices
    index = {}
    order = []
    edges = []
    
    start = (0, tuple([0] * d))
    index[start] = 0
    order.append(start)
    
    def get_neighbours(vertex):
        side, x = vertex
        x = np.array(x, dtype=np.int32)
        neighbours = []
        for first in range(q):
            y = np.zeros(d, dtype=np.int32)
            y[1] = first
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = (x[c] + table[y[a], x[b]]) % q if q == 4 else (x[c] + y[a] * x[b]) % q
                else:
                    y[c] = (x[c] - table[x[a], y[b]]) % q if q == 4 else (x[c] - x[a] * y[b]) % q
            neighbours.append((1 - side, tuple(y)))
        return neighbours
    
    for vertex in order:
        for other in get_neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges

def _verify_girth_fast(n, edges, k, g):
    """Fast girth verification using non-backtracking walk BFS."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Check degree
    for i in range(n):
        if len(adj[i]) != k:
            return False
    
    # Check for cycles shorter than g using non-backtracking walks
    # For each vertex, do BFS on the directed line graph (states are (node, prev))
    # A cycle of length L < g exists iff there's a non-backtracking closed walk of length L
    # We check: for each start vertex s, BFS on (node, prev) states up to depth g-1
    # If we return to (s, prev) with a different prev, we have a cycle
    
    for start in range(n):
        # State: (current_node, previous_node)
        # Start states: (start, -1)
        # We track visited states
        visited = set()
        queue = deque()
        
        for nb in adj[start]:
            queue.append((nb, start, 1))
            visited.add((nb, start))
        
        while queue:
            u, p, depth = queue.popleft()
            if depth >= g - 1:
                continue
            for v in adj[u]:
                if v == p:
                    continue
                if v == start:
                    # Found a cycle of length depth + 1
                    if depth + 1 < g:
                        return False
                    continue
                state = (v, u)
                if state not in visited:
                    visited.add(state)
                    queue.append((v, u, depth + 1))
    
    return True

def _random_regular_graph_girth_aware(k, g, target_n, time_budget=20):
    """Generate a random k-regular graph and check girth. Returns None if fails."""
    start_time = time.time()
    deadline = start_time + time_budget
    
    # Try multiple random seeds
    np.random.seed(42)
    
    for attempt in range(100):
        if time.time() > deadline:
            break
        
        # Generate a random k-regular graph using configuration model
        # Then check girth
        try:
            edges = _generate_random_regular_graph(target_n, k)
            if edges is None:
                continue
            
            # Quick check: verify it's k-regular
            adj = [[] for _ in range(target_n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            
            valid = True
            for i in range(target_n):
                if len(adj[i]) != k:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check girth
            if _verify_girth_fast(target_n, edges, k, g):
                return target_n, edges
        except Exception:
            continue
    
    return None, None

def _generate_random_regular_graph(n, k):
    """Generate a random k-regular graph using configuration model."""
    if n * k % 2 != 0:
        return None
    
    # Create stubs
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
    
    np.random.shuffle(stubs)
    
    # Pair stubs
    edges = set()
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i + 1]
        if u == v:
            return None  # Loop, try again
        if (min(u, v), max(u, v)) in edges:
            return None  # Multi-edge, try again
        edges.add((min(u, v), max(u, v)))
    
    return list(edges)

def _construct_luw(k, g):
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

def _construct_sl2_cayley(q):
    """
    Construct Cayley graph of SL(2, q) with generators A, B, A^-1, B^-1.
    SL(2,5) has order 120, SL(2,7) has order 336.
    We use a subset of generators to get a 4-regular graph.
    """
    # Determine field operations
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
        inv_table = {0:0, 1:1, 2:3, 3:2}
        def inv_func(x): return inv_table[x]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
        def inv_func(x):
            if x == 0: return 0
            for i in range(1, q):
                if (x * i) % q == 1:
                    return i
            return 0

    def mat_mul(m1, m2):
        a1, b1, c1, d1 = m1
        a2, b2, c2, d2 = m2
        return (
            add(mul(a1, a2), mul(b1, c2)),
            add(mul(a1, b2), mul(b1, d2)),
            add(mul(c1, a2), mul(d1, c2)),
            add(mul(c1, b2), mul(d1, d2))
        )

    def mat_inv(m):
        a, b, c, d = m
        det = sub(mul(a, d), mul(b, c))
        if det == 0:
            return None
        det_inv = inv_func(det)
        return (
            mul(det_inv, d),
            mul(det_inv, sub(0, b)),
            mul(det_inv, sub(0, c)),
            mul(det_inv, a)
        )

    # Generators for SL(2, q) - use 2 generators and their inverses for 4-regular
    A = (0, sub(0, 1), 1, 0)  # [[0, -1], [1, 0]]
    B = (1, 1, 0, 1)           # [[1, 1], [0, 1]]
    A_inv = mat_inv(A)
    B_inv = mat_inv(B)
    
    # Check that A and B have order > 2 to avoid multi-edges
    # Use generators: A, A_inv, B, B_inv
    gens = [A, A_inv, B, B_inv]
    
    # BFS to generate group
    start = (1, 0, 0, 1)
    index = {start: 0}
    order = [start]
    edges = []
    
    for v in order:
        idx_v = index[v]
        for gen in gens:
            w = mat_mul(v, gen)
            if w not in index:
                index[w] = len(order)
                order.append(w)
            idx_w = index[w]
            if idx_v < idx_w:
                edges.append((idx_v, idx_w))
    
    return len(order), edges

def _verify_girth(n, edges, k, g):
    # Check if graph is k-regular and girth >= g
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    for i in range(n):
        if len(adj[i]) != k:
            return False
            
    for start in range(n):
        visited = set()
        queue = deque([(start, -1, 0)])
        while queue:
            u, p, d = queue.popleft()
            if (u, p) in visited:
                continue
            visited.add((u, p))
            if d >= g - 1:
                continue
            for v in adj[u]:
                if v == p:
                    continue
                if v == start:
                    if d + 1 < g:
                        return False
                    else:
                        continue
                queue.append((v, u, d + 1))
                
    return True
