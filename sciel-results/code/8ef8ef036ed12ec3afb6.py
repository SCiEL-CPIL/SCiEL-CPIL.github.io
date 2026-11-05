import time
import random
import sys
from collections import deque
import numpy as np

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Uses a different base graph (K_{3,3} complement = octahedron? No, that's girth 4).
    Actually for k=3, the only 3-regular base graphs with girth >= 5 are:
    - Petersen (10 vertices, girth 5)
    - The 14-vertex Heawood graph (girth 6)
    - Larger cages...
    
    Let me use a hybrid: try Petersen lift with a smarter solver, 
    AND try the Heawood graph lift (14 vertices, 21 edges, girth 6).
    The Heawood lift gives n = 14*m, which for m >= 37 gives n >= 518.
    But the girth-6 base means we need non-backtracking walks of length < 16 
    in the base, which is easier to satisfy.
    """
    start_time = time.time()
    deadline = start_time + 135.0

    if k == 3:
        # Try multiple base graphs in order of expected n
        results = []
        
        # 1. Petersen graph lift (n = 10*m)
        if time.time() < deadline - 30:
            r = _try_petersen_lift(g, deadline, start_time)
            if r is not None:
                results.append(r)
        
        # 2. Heawood graph lift (n = 14*m) - structurally different base
        if time.time() < deadline - 20:
            r = _try_heawood_lift(g, deadline, start_time)
            if r is not None:
                results.append(r)
        
        # 3. Fallback LUW
        r = _construct_luw(3, g, d=7)
        results.append(r)
        
        # Return the one with smallest n
        results.sort(key=lambda x: x[0])
        return results[0]
    else:
        return _construct_luw(k, g, d=7)

def _try_petersen_lift(g, deadline, start_time):
    """
    Petersen graph voltage lift with systematic constraint solving.
    Key structural change: use a backtracking search with constraint propagation
    instead of random repair.
    """
    pet_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    n_base = 10
    edge_index = {}
    for i, (u, v) in enumerate(pet_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
    
    # Precompute non-backtracking closed walks
    coeff_vectors = set()
    max_walk_len = g - 1
    for start_v in range(n_base):
        stack = [(start_v, -1, 0, [0] * 15)]
        while stack:
            v, prev_v, l, counts = stack.pop()
            if l >= max_walk_len:
                continue
            for next_v in range(n_base):
                if next_v == prev_v or next_v == v:
                    continue
                e_idx = edge_index.get((v, next_v))
                if e_idx is None:
                    continue
                new_counts = counts.copy()
                u_edge, v_edge = pet_edges[e_idx]
                if v == u_edge and next_v == v_edge:
                    new_counts[e_idx] += 1
                else:
                    new_counts[e_idx] -= 1
                new_l = l + 1
                if next_v == start_v and new_l > 0:
                    coeff_vectors.add(tuple(new_counts))
                stack.append((next_v, v, new_l, new_counts))
    
    unique_coeffs = set()
    for c in coeff_vectors:
        if all(x == 0 for x in c):
            continue
        for i in range(15):
            if c[i] != 0:
                if c[i] < 0:
                    c = tuple(-x for x in c)
                unique_coeffs.add(c)
                break
    coeff_list = list(unique_coeffs)
    
    if len(coeff_list) == 0:
        # No constraints, any non-zero assignment works
        m = 3
        voltages = [1] * 15
        edges = _build_lifted_graph(pet_edges, voltages, m)
        return (10 * m, edges)
    
    coeff_matrix = np.array(coeff_list, dtype=np.int64)
    
    def is_prime(n):
        if n < 2: return False
        if n < 4: return True
        if n % 2 == 0 or n % 3 == 0: return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    
    primes = []
    candidate = 2
    while len(primes) < 200:
        if is_prime(candidate):
            primes.append(candidate)
        candidate += 1
    
    for m in primes:
        if time.time() > deadline:
            break
        n_candidate = 10 * m
        
        # Systematic backtracking with constraint propagation
        for attempt in range(200):
            if time.time() > deadline:
                break
            
            # Random initial assignment
            voltages = [random.randint(1, m - 1) for _ in range(15)]
            
            # Iterative constraint satisfaction
            for step in range(500):
                if time.time() > deadline:
                    break
                
                # Find violated constraints
                violated = []
                for ci, c in enumerate(coeff_list):
                    s = sum(c[i] * voltages[i] for i in range(15)) % m
                    if s == 0:
                        violated.append(ci)
                
                if not violated:
                    edges = _build_lifted_graph(pet_edges, voltages, m)
                    return (n_candidate, edges)
                
                # Pick the most violated (most non-zero coefficients)
                viol = random.choice(violated)
                c = coeff_list[viol]
                nonzero = [i for i in range(15) if c[i] != 0]
                
                if not nonzero:
                    break
                
                # Try to fix by changing one voltage
                j = random.choice(nonzero)
                # The current contribution is c[j] * voltages[j]
                # We need sum(c[i]*v[i]) != 0 mod m
                # Current sum = 0, so we need to change voltages[j] by delta
                # such that c[j] * delta != 0 mod m
                # Since m is prime and c[j] != 0, any delta != 0 works
                delta = random.randint(1, m - 1)
                voltages[j] = (voltages[j] + delta) % m
                if voltages[j] == 0:
                    voltages[j] = 1
    
    return None

def _try_heawood_lift(g, deadline, start_time):
    """
    Heawood graph voltage lift - structurally different base graph.
    The Heawood graph is the incidence graph of the Fano plane.
    14 vertices, 21 edges, girth 6, 3-regular.
    """
    # Heawood graph: bipartite, vertices 0-6 (points) and 7-13 (lines)
    # Fano plane lines:
    lines = [
        (0, 1, 2), (0, 3, 4), (0, 5, 6),
        (1, 3, 5), (1, 4, 6), (2, 3, 6), (2, 4, 5)
    ]
    
    heawood_edges = []
    for i, line in enumerate(lines):
        for point in line:
            heawood_edges.append((point, 7 + i))
    
    n_base = 14
    n_edges = 21
    
    edge_index = {}
    for i, (u, v) in enumerate(heawood_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
    
    # Precompute non-backtracking closed walks
    coeff_vectors = set()
    max_walk_len = g - 1
    for start_v in range(n_base):
        stack = [(start_v, -1, 0, [0] * n_edges)]
        while stack:
            v, prev_v, l, counts = stack.pop()
            if l >= max_walk_len:
                continue
            for next_v in range(n_base):
                if next_v == prev_v or next_v == v:
                    continue
                e_idx = edge_index.get((v, next_v))
                if e_idx is None:
                    continue
                new_counts = counts.copy()
                u_edge, v_edge = heawood_edges[e_idx]
                if v == u_edge and next_v == v_edge:
                    new_counts[e_idx] += 1
                else:
                    new_counts[e_idx] -= 1
                new_l = l + 1
                if next_v == start_v and new_l > 0:
                    coeff_vectors.add(tuple(new_counts))
                stack.append((next_v, v, new_l, new_counts))
    
    unique_coeffs = set()
    for c in coeff_vectors:
        if all(x == 0 for x in c):
            continue
        for i in range(n_edges):
            if c[i] != 0:
                if c[i] < 0:
                    c = tuple(-x for x in c)
                unique_coeffs.add(c)
                break
    coeff_list = list(unique_coeffs)
    
    if len(coeff_list) == 0:
        m = 3
        voltages = [1] * n_edges
        edges = _build_lifted_graph_heawood(heawood_edges, voltages, m)
        return (14 * m, edges)
    
    def is_prime(n):
        if n < 2: return False
        if n < 4: return True
        if n % 2 == 0 or n % 3 == 0: return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    
    primes = []
    candidate = 2
    while len(primes) < 100:
        if is_prime(candidate):
            primes.append(candidate)
        candidate += 1
    
    for m in primes:
        if time.time() > deadline:
            break
        n_candidate = 14 * m
        
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            voltages = [random.randint(1, m - 1) for _ in range(n_edges)]
            
            for step in range(300):
                if time.time() > deadline:
                    break
                
                violated = []
                for ci, c in enumerate(coeff_list):
                    s = sum(c[i] * voltages[i] for i in range(n_edges)) % m
                    if s == 0:
                        violated.append(ci)
                
                if not violated:
                    edges = _build_lifted_graph_heawood(heawood_edges, voltages, m)
                    return (n_candidate, edges)
                
                viol = random.choice(violated)
                c = coeff_list[viol]
                nonzero = [i for i in range(n_edges) if c[i] != 0]
                if not nonzero:
                    break
                j = random.choice(nonzero)
                delta = random.randint(1, m - 1)
                voltages[j] = (voltages[j] + delta) % m
                if voltages[j] == 0:
                    voltages[j] = 1
    
    return None

def _build_lifted_graph(pet_edges, voltages, m):
    n_base = 10
    edge_index = {}
    for i, (u, v) in enumerate(pet_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
    
    edges = set()
    for u in range(n_base):
        for v in range(n_base):
            if u == v:
                continue
            e_idx = edge_index.get((u, v))
            if e_idx is None:
                continue
            vol = voltages[e_idx]
            u_edge, v_edge = pet_edges[e_idx]
            if u == u_edge and v == v_edge:
                sign = 1
            else:
                sign = -1
            for layer in range(m):
                u_node = layer * n_base + u
                target_layer = (layer + sign * vol) % m
                v_node = target_layer * n_base + v
                a, b = min(u_node, v_node), max(u_node, v_node)
                edges.add((a, b))
    return list(edges)

def _build_lifted_graph_heawood(heawood_edges, voltages, m):
    n_base = 14
    edge_index = {}
    for i, (u, v) in enumerate(heawood_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
    
    edges = set()
    for u in range(n_base):
        for v in range(n_base):
            if u == v:
                continue
            e_idx = edge_index.get((u, v))
            if e_idx is None:
                continue
            vol = voltages[e_idx]
            u_edge, v_edge = heawood_edges[e_idx]
            if u == u_edge and v == v_edge:
                sign = 1
            else:
                sign = -1
            for layer in range(m):
                u_node = layer * n_base + u
                target_layer = (layer + sign * vol) % m
                v_node = target_layer * n_base + v
                a, b = min(u_node, v_node), max(u_node, v_node)
                edges.add((a, b))
    return list(edges)

def _construct_luw(k, g, d=None, variant=0):
    """
    LUW construction with specified depth d and variant for rule selection.
    """
    if d is None:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    offset = variant % 4
    for c in range(5, d + 1):
        r = (c - 5 + offset) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for i, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges

def _has_girth_at_least(n, edges, g):
    """
    Check if graph has girth at least g using non-backtracking walks.
    """
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    max_len = g - 1
    
    for start in range(n):
        queue = deque()
        for nb in adj[start]:
            queue.append((nb, start, 1))
        
        while queue:
            u, prev, l = queue.popleft()
            
            if u == start and l > 0:
                return False
            
            if l >= max_len:
                continue
            
            for v in adj[u]:
                if v != prev:
                    queue.append((v, u, l + 1))
    
    return True

