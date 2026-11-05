import time
import random
import sys
from collections import deque
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 135.0

    if k == 3:
        # Try direct CSP approach first (can find smaller graphs)
        result = _construct_direct_csp(3, g, deadline, start_time)
        if result is not None:
            return result
        # Fallback to Petersen voltage lift
        return _construct_petersen_lift(g, deadline, start_time)
    else:
        return _construct_luw(k, g, d=7)

def _construct_direct_csp(k, g, deadline, start_time):
    """
    Direct tree-embedding CSP for k=3, g=16.
    Build a depth-7 tree (1+3+9+27+81+243+729+2187=3280 vertices),
    then close the boundary with 3-regularity and girth constraints.
    """
    if g < 14 or g > 18:
        return None
    
    # Moore bound for k=3, g=16: 1 + 3*(1 + 2 + 4 + 8 + 16 + 32 + 64) = 1 + 3*127 = 382
    # Actually for girth 16 (even), Moore bound = 1 + k*(k-1)^{g/2-1} + (k-1)*((k-1)^{g/2-1} - 1)/(k-2)
    # For k=3, g=16: 1 + 3*2^7 + 2*(2^7-1) = 1 + 384 + 254 = 639? No...
    # Moore bound for cage n(3,16): 1 + 3*(2^7) = 1 + 384 = 385? 
    # Actually: for even girth g, n >= 1 + k * sum_{i=0}^{g/2-1} (k-1)^i
    # = 1 + 3 * (1 + 2 + 4 + 8 + 16 + 32 + 64) = 1 + 3*127 = 382
    
    # The tree has depth d = g/2 - 1 = 7
    # Vertices: 1 + 3 + 9 + 27 + 81 + 243 + 729 + 2187 = 3280
    # Boundary (depth 7): 2187 vertices, each needs 2 more edges (degree 3, one used for parent)
    # Total boundary stubs: 2187 * 2 = 4374
    # We need to pair these stubs into 2187 edges
    # But we must avoid creating cycles shorter than g=16
    
    # This is a massive CSP. Let's use a smarter approach:
    # Instead of full tree, use a smaller structure and add edges.
    
    # Alternative: Start with a random 3-regular graph near the Moore bound
    # and try to increase girth via edge swaps.
    
    # For now, try a randomized approach:
    # 1. Generate random 3-regular graphs of size n
    # 2. Check girth
    # 3. Try edge swaps to increase girth
    
    # Moore bound for (3,16) is 382, best known is 936
    # Let's try sizes from 382 to 936
    
    # Actually, let's try a more systematic approach:
    # Use the known structure: start with a tree of appropriate depth,
    # then close it with a matching on the boundary.
    
    # For practical purposes within time limits, let's try:
    # 1. Small random 3-regular graphs with girth checking
    # 2. If we find one with girth >= g, return it
    
    # Generate random 3-regular graphs and check girth
    best_n = float('inf')
    best_edges = None
    
    # Try various sizes
    for n in [382, 384, 386, 388, 390, 392, 394, 396, 398, 400,
              402, 404, 406, 408, 410, 412, 414, 416, 418, 420,
              422, 424, 426, 428, 430, 432, 434, 436, 438, 440,
              442, 444, 446, 448, 450, 452, 454, 456, 458, 460,
              462, 464, 466, 468, 470, 472, 474, 476, 478, 480,
              482, 484, 486, 488, 490, 492, 494, 496, 498, 500]:
        
        if time.time() > deadline - 30:  # Leave time for fallback
            break
        
        if n >= best_n:
            continue
        
        # n must be even for 3-regular graph
        if n % 2 != 0:
            continue
        
        # Try to construct a 3-regular graph of size n with girth >= g
        # Use random generation with girth checking
        found = False
        max_attempts = 50
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 30:
                break
            
            edges = _random_3_regular(n)
            if edges is None:
                continue
            
            if _has_girth_at_least(n, edges, g):
                best_n = n
                best_edges = edges
                found = True
                break
        
        if found:
            return best_n, best_edges
    
    return None

def _random_3_regular(n):
    """Generate a random 3-regular graph on n vertices."""
    if n % 2 != 0 or n < 4:
        return None
    
    # Use configuration model
    stubs = []
    for i in range(n):
        for _ in range(3):
            stubs.append(i)
    
    random.shuffle(stubs)
    
    # Pair up stubs
    edges = set()
    valid = True
    for i in range(0, n * 3, 2):
        u, v = stubs[i], stubs[i + 1]
        if u == v:
            valid = False
            break
        if (u, v) in edges or (v, u) in edges:
            valid = False
            break
        edges.add((min(u, v), max(u, v)))
    
    if not valid:
        return None
    
    # Verify all vertices have degree 3
    degree = [0] * n
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
    
    if any(d != 3 for d in degree):
        return None
    
    return list(edges)

def _has_girth_at_least(n, edges, g):
    """
    Check if graph has girth at least g using non-backtracking walks.
    Uses BFS from each vertex.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    max_len = g - 1
    
    for start in range(n):
        # BFS with non-backtracking constraint
        # State: (vertex, previous_vertex, length)
        visited = set()
        queue = deque()
        
        for nb in adj[start]:
            queue.append((nb, start, 1))
        
        while queue:
            u, prev, l = queue.popleft()
            
            if u == start:
                if l < g:
                    return False
                continue
            
            if l >= max_len:
                continue
            
            for v in adj[u]:
                if v != prev:
                    state = (v, u, l + 1)
                    if state not in visited:
                        visited.add(state)
                        queue.append(state)
    
    return True

def _construct_petersen_lift(g, deadline, start_time):
    """
    Constructs a girth-g graph by lifting the Petersen graph with voltages in Z_m.
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
    
    # Precompute all non-backtracking closed walks of length < g in the Petersen graph
    coeff_vectors = set()
    max_walk_len = g - 1
    
    for start_v in range(n_base):
        stack = [(start_v, -1, 0, [0]*15)]
        
        while stack:
            v, prev_v, l, counts = stack.pop()
            
            if l >= max_walk_len:
                continue
            
            for next_v in range(n_base):
                if next_v == prev_v:
                    continue
                if next_v == v:
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
    
    # Normalize constraints
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
    
    if not coeff_list:
        voltages = [1] * 15
        m = 2
        n_candidate = 10 * m
        edges = _build_lifted_graph(pet_edges, voltages, m)
        return n_candidate, edges
    
    coeff_matrix = np.array(coeff_list, dtype=np.int64)
    
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
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
    
    best_n = float('inf')
    
    for m in primes:
        if time.time() > deadline:
            break
        
        n_candidate = 10 * m
        if n_candidate >= best_n:
            continue
        
        max_attempts = 1200
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            voltages = np.random.randint(1, m, size=15).astype(np.int64)
            
            for repair_step in range(120):
                if time.time() > deadline:
                    break
                
                products = coeff_matrix @ voltages
                residues = np.mod(products, m)
                violated_indices = np.where(residues == 0)[0]
                
                if len(violated_indices) == 0:
                    voltages_list = voltages.tolist()
                    edges = _build_lifted_graph(pet_edges, voltages_list, m)
                    return n_candidate, edges
                
                violated_idx = violated_indices[random.randint(0, len(violated_indices)-1)]
                violated_c = coeff_matrix[violated_idx]
                nonzero_indices = np.where(violated_c != 0)[0]
                if len(nonzero_indices) == 0:
                    break
                j = nonzero_indices[random.randint(0, len(nonzero_indices)-1)]
                
                current_sum = int(np.dot(violated_c, voltages)) % m
                c_j = int(violated_c[j])
                
                fixed = False
                for _ in range(8):
                    new_val = random.randint(1, m-1)
                    if new_val == int(voltages[j]):
                        continue
                    new_sum = (current_sum + c_j * (new_val - int(voltages[j]))) % m
                    if new_sum != 0:
                        voltages[j] = new_val
                        fixed = True
                        break
                
                if not fixed:
                    voltages[j] = random.randint(1, m-1)
    
    return _construct_luw(k=3, g=g, d=11)

def _build_lifted_graph(pet_edges, voltages, m):
    n_base = 10
    edge_index = {}
    for i, (u, v) in enumerate(pet_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
    
    edges = []
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
                
                if u_node < v_node:
                    edges.append((u_node, v_node))
                else:
                    edges.append((v_node, u_node))
                
    edges = list(set(edges))
    return edges

def _construct_luw(k, g, d=None, variant=0):
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
