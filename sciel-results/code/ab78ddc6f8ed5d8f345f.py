import time
import math
from collections import deque
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    best_n = float('inf')
    best_edges = None
    
    # Candidate 1: Lazebnik-Ustimenko-Woldar (LUW) - vectorized
    try:
        n_luw, edges_luw = _construct_luw_vectorized(k, g)
        if _verify_girth_fast(n_luw, edges_luw, k, g):
            if n_luw < best_n:
                best_n = n_luw
                best_edges = edges_luw
    except Exception:
        pass
    
    # Candidate 2: Probabilistic random regular graph with girth-aware construction
    if k == 4 and time.time() < deadline - 30:
        try:
            # Try to find a smaller graph using random regular graph generation
            # The Moore bound for (4,10) is 2*(1+3*(3^8-1)/2) = 2*(1+3*6560/2) = 2*(1+9840) = 19682? No.
            # Moore bound: n >= 2 * (d-1) * sum_{i=0}^{floor((g-1)/2)-1} (d-1)^i for even g
            # For k=4, g=10: n >= 2 * 3 * (1 + 3 + 9 + 27 + 81) = 2*3*121 = 726
            # Wait, let me recalculate. Moore bound for (k,g):
            # If g is even: n >= k * sum_{i=0}^{g/2-1} (k-1)^i
            # For k=4, g=10: n >= 4 * (1 + 3 + 9 + 27 + 81) = 4 * 121 = 484
            # So we need at least 484 vertices. The LUW construction gives some specific n.
            
            # Try random regular graphs starting from a reasonable size
            # We know the best known is 320, so let's try sizes between 320 and the LUW result
            if best_n > 320:
                for trial_n in [320, 324, 328, 332, 336, 340]:
                    if trial_n >= best_n:
                        break
                    if time.time() > deadline - 10:
                        break
                    try:
                        n_rand, edges_rand = _random_regular_graph_girth_aware(k, g, trial_n, time_budget=20)
                        if n_rand == trial_n and _verify_girth_fast(n_rand, edges_rand, k, g):
                            best_n = n_rand
                            best_edges = edges_rand
                            break
                    except Exception:
                        pass
        except Exception:
            pass
    
    # If no graph found, fallback to LUW
    if best_edges is None:
        try:
            best_n, best_edges = _construct_luw_vectorized(k, g)
        except Exception:
            # Fallback to original LUW
            try:
                best_n, best_edges = _construct_luw(k, g)
            except Exception:
                best_n = k
                best_edges = []
                
    return best_n, best_edges

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
    # Vertex representation: (side, array of d values)
    # We'll use a dict mapping (side, tuple) to index
    
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
    """Fast girth verification using adjacency lists and BFS."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Check degree
    for i in range(n):
        if len(adj[i]) != k:
            return False
            
    # Check for cycles shorter than g using BFS from each vertex
    # We only need to check up to depth g-1
    for start in range(n):
        # BFS with parent tracking
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            du = dist[u]
            if du >= g - 1:
                continue
            for v in adj[u]:
                if v == parent[u]:
                    continue
                if dist[v] == -1:
                    dist[v] = du + 1
                    parent[v] = u
                    queue.append(v)
                elif dist[v] <= du:
                    # Found a cycle
                    # Cycle length is at least dist[u] + 1 + dist[v] - 2*dist[lca]
                    # But we can detect if there's a cycle of length < g
                    # If dist[u] + 1 + dist[v] < g, then we have a cycle shorter than g
                    # Actually, this isn't quite right. Let me use a different approach.
                    pass
        
        # Alternative: check if any two neighbors of start are connected by a path of length < g-2
        # This is more complex. Let's use the standard non-backtracking walk check.
        # For efficiency, we'll do a BFS that tracks (node, prev) states
        
    # More efficient check: for each edge (u,v), check if there's a path from u to v of length < g-1 that doesn't use edge (u,v)
    # This would indicate a cycle of length < g
    
    # Let's use a simpler approach: BFS from each vertex, and if we find a back-edge to a vertex at distance d where current_dist + 1 + d < g, we have a short cycle
    # Actually, the standard way is:
    # For each vertex s, do BFS. If we find an edge (u,v) where both u and v are visited and dist[u] + 1 + dist[v] < g (and the path doesn't go through the same edge), we have a cycle.
    
    # Let me implement a proper check:
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            du = dist[u]
            if du >= g - 1:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    queue.append(v)
                else:
                    # Back edge or cross edge
                    # The cycle length is at least min(du + 1 + dist[v], ...) but this is an upper bound
                    # Actually, for undirected graphs, if we find an edge (u,v) where both are visited,
                    # the cycle length is at most du + 1 + dist[v] (going from start to u, then u-v, then v back to start)
                    # But this might not be the shortest cycle.
                    # A better check: if du + 1 + dist[v] < g, we definitely have a cycle of length < g
                    # (the path from start to u has length du, edge u-v, path from v to start has length dist[v])
                    # But wait, the path from v to start might share vertices with the path from start to u.
                    # In a BFS tree, the paths are simple, so the cycle formed is simple.
                    # The cycle length is du + 1 + dist[v] only if the paths are vertex-disjoint except at start.
                    # In general, the cycle length is at most du + 1 + dist[v].
                    if du + 1 + dist[v] < g:
                        return False
    
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
    # Construct Cayley graph of SL(2, q) with generators A, B, A^-1, B^-1
    # Matrices are (a, b, c, d) representing [[a, b], [c, d]]
    # Identity: (1, 0, 0, 1)
    
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

    # Generators for SL(2, q)
    A = (0, sub(0, 1), 1, 0)
    B = (1, 1, 0, 1)
    A_inv = mat_inv(A)
    B_inv = mat_inv(B)
    
    gens = [A, B, A_inv, B_inv]
    
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
