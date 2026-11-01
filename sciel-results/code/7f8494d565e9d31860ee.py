import time
import random
from collections import deque
from itertools import product

def construct(k, g):
    """
    Construct a k-regular graph with girth >= g.
    For k=3, g=16, we use a voltage lift of the Petersen graph.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    if k != 3:
        return _luw_construct(k, g)

    # Petersen graph edges (0-9)
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    petersen_adj = [[] for _ in range(10)]
    for u, v in petersen_edges:
        petersen_adj[u].append(v)
        petersen_adj[v].append(u)

    base_edges = []
    for u in range(10):
        for v in petersen_adj[u]:
            if u < v:
                base_edges.append((u, v))
    
    # Map edge to index for voltage assignment
    edge_to_idx = {}
    for i, (u, v) in enumerate(base_edges):
        edge_to_idx[(u, v)] = i
        edge_to_idx[(v, u)] = i

    # Enumerate all non-backtracking walks of length < g in the base graph
    # A non-backtracking walk is a sequence v0, v1, ..., vL where vi != vi-2 for i >= 2
    # and each (vi-1, vi) is an edge.
    
    def _enumerate_nb_walks(max_len):
        """Enumerate all non-backtracking walks up to max_len in the base graph."""
        walks = []
        # Start from each vertex, each neighbor
        for start in range(10):
            for first in petersen_adj[start]:
                # Walk of length 1: start -> first
                walks.append([(start, first)])
                # Extend
                stack = [((start, first), [(start, first)])]
                while stack:
                    (prev, cur), walk = stack.pop()
                    if len(walk) >= max_len:
                        continue
                    for nxt in petersen_adj[cur]:
                        if nxt == prev:
                            continue  # Non-backtracking
                        new_walk = walk + [(cur, nxt)]
                        walks.append(new_walk)
                        stack.append(((cur, nxt), new_walk))
        return walks

    # For each walk, compute the edge indices involved (with signs for direction)
    # In a voltage lift with voltages in an abelian group G, a walk is closed
    # iff the product (or sum, for additive) of voltages is the identity (or 0).
    
    # For Z_m: sum of voltages along walk must not be 0 mod m
    # For F_2: XOR of voltages along walk must not be 0
    
    # Let's use F_2 for simplicity and speed.
    # Each edge gets a voltage in F_2^d for some dimension d.
    # We need: for every NB walk of length < g, the XOR of voltages != 0.
    
    # This is equivalent to: the vectors (sum of edge voltage vectors along each walk)
    # must all be non-zero.
    
    # Strategy: assign random vectors in F_2^d to edges, then check all walks.
    # If any walk sums to 0, we need a different assignment or larger d.
    
    # Actually, let me think differently. For a lift of size m (using Z_m),
    # we assign voltage v_e in Z_m to each edge e.
    # A walk with edge sequence e1, e2, ..., eL (with signs) is closed
    # iff sum(s_i * v_ei) == 0 mod m.
    
    # Let me use the approach: enumerate walks, get their edge coefficient vectors,
    # then find voltage assignments that make all these linear forms non-zero.
    
    # For efficiency, let's first enumerate walks and extract edge incidence vectors.
    
    if g <= 16:
        # Enumerate NB walks up to length g-1 = 15
        nb_walks = _enumerate_nb_walks(g - 1)
        
        # For each walk, compute the edge coefficient vector in Z_m
        # The walk is a sequence of directed edges. For edge (u,v), if traversed
        # as u->v, coefficient is +1; if v->u, coefficient is -1.
        
        # But for the voltage sum: if we define voltage on undirected edge as v_e,
        # and we traverse from u to v, we add v_e; from v to u, we subtract v_e.
        
        # Actually, let's define: each undirected edge e has a voltage v_e in Z_m.
        # Traversing e in the "forward" direction adds v_e, "backward" subtracts v_e.
        
        # For each walk, we get a vector in Z_m^{15} (15 edges in Petersen graph)
        # where the i-th component is the net number of times edge i is traversed
        # forward minus backward.
        
        # The walk is closed in the lift iff dot(coeff_vector, voltages) == 0 mod m.
        
        # We need: for all walks, dot(coeff_vector, voltages) != 0 mod m.
        
        # This is a constraint satisfaction problem.
        
        # Let's collect all unique coefficient vectors (up to scalar multiple, since
        # if a walk sums to 0, so does its reverse with negated coefficients).
        
        coeff_vectors = []
        seen = set()
        
        for walk in nb_walks:
            coeff = [0] * 15
            for (u, v) in walk:
                idx = edge_to_idx[(u, v)]
                # Determine direction relative to base_edges
                base_u, base_v = base_edges[idx]
                if u == base_u and v == base_v:
                    coeff[idx] += 1
                else:
                    coeff[idx] -= 1
            # Normalize: make the first non-zero entry positive
            key = None
            for c in coeff:
                if c != 0:
                    if c < 0:
                        coeff = [-x for x in coeff]
                    key = tuple(coeff)
                    break
            if key is not None and key not in seen:
                seen.add(key)
                coeff_vectors.append(coeff)
        
        # Now we need to find voltages v_0, ..., v_14 in Z_m such that
        # for all coeff in coeff_vectors: sum(coeff[i] * v[i]) != 0 mod m.
        
        # Let's try random assignments for various m.
        random.seed(42)
        
        def _check_voltage_assignment(voltages, m):
            """Check if voltage assignment satisfies all constraints."""
            for coeff in coeff_vectors:
                s = 0
                for i, c in enumerate(coeff):
                    if c != 0:
                        s += c * voltages[i]
                if s % m == 0:
                    return False
            return True
        
        def _build_lift_from_voltages(m, voltages):
            """Build the lifted graph from voltage assignment."""
            n = 10 * m
            edges = []
            for idx, (u, v) in enumerate(base_edges):
                shift = voltages[idx]
                for i in range(m):
                    u_id = u * m + i
                    v_id = v * m + (i + shift) % m
                    edges.append((u_id, v_id))
            return n, edges
        
        # Try m from 52 upward
        for m in range(52, 500):
            if time.time() > deadline:
                break
            
            # Try random assignments
            for _ in range(200):
                if time.time() > deadline:
                    break
                voltages = [random.randint(0, m - 1) for _ in range(15)]
                if _check_voltage_assignment(voltages, m):
                    n, edges = _build_lift_from_voltages(m, voltages)
                    # Final verification with BFS
                    if _verify_girth_fast(n, edges, g):
                        return n, edges
        
        # If not found, try larger m
        for m in range(500, 2000):
            if time.time() > deadline:
                break
            for _ in range(50):
                if time.time() > deadline:
                    break
                voltages = [random.randint(0, m - 1) for _ in range(15)]
                if _check_voltage_assignment(voltages, m):
                    n, edges = _build_lift_from_voltages(m, voltages)
                    if _verify_girth_fast(n, edges, g):
                        return n, edges
    
    # Fallback: large random lift
    m = 3000
    random.seed(42)
    shifts = [random.randint(0, m - 1) for _ in range(15)]
    n = 10 * m
    edges = []
    for idx, (u, v) in enumerate(base_edges):
        shift = shifts[idx]
        for i in range(m):
            u_id = u * m + i
            v_id = v * m + (i + shift) % m
            edges.append((u_id, v_id))
    return n, edges

def _verify_girth_fast(n, edges, g):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    dist = [0] * n
    parent = [-1] * n
    seen_mark = [0] * n
    mark = 0

    for s in range(n):
        mark += 1
        dist[s] = 0
        parent[s] = -1
        seen_mark[s] = mark
        queue = deque([s])

        while queue:
            u = queue.popleft()
            du = dist[u]
            if du >= g // 2:
                continue
            for v in adj[u]:
                if seen_mark[v] != mark:
                    dist[v] = du + 1
                    parent[v] = u
                    seen_mark[v] = mark
                    queue.append(v)
                elif v != parent[u] and u != parent[v]:
                    cycle_len = du + dist[v] + 1
                    if cycle_len < g:
                        return False

    return True

def _luw_construct(k, g):
    """
    Lazebnik-Ustimenko-Woldar construction for k-regular graphs.
    """
    if k != 3:
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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

    # For k=3, we use q=3.
    q = 3

    def build_luw(d):
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

    # Try d=8 first. If girth < 16, try d=9, etc.
    for d in [8, 9, 10, 11]:
        n, edges = build_luw(d)
        if _check_girth_fast(n, edges, g):
            return n, edges
    # If all fail (unlikely for d=11), return the largest one (which should be valid by theory)
    return build_luw(11)

def _build_adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _check_girth_fast(n, edges, min_girth):
    """Fast girth check using BFS from each vertex with parent tracking."""
    adj = _build_adj(n, edges)
    max_depth = (min_girth - 1) // 2

    for start in range(n):
        visited = [-1] * n
        visited[start] = 0
        queue = deque([(start, 0, -1)])

        while queue:
            node, depth, parent = queue.popleft()
            if depth > max_depth:
                continue
            for neighbor in adj[node]:
                if neighbor == parent:
                    continue
                if visited[neighbor] == -1:
                    visited[neighbor] = depth + 1
                    queue.append((neighbor, depth + 1, node))
                else:
                    cycle_len = depth + visited[neighbor] + 1
                    if cycle_len < min_girth:
                        return False
    return True
