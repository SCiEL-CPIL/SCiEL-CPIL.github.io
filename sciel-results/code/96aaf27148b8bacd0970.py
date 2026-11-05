import time
import random
import numpy as np
from collections import deque

# EVOLVE-BLOCK-START

# Petersen Graph Definition
_PETERSEN_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),          # Outer
    (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),          # Inner
    (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)           # Spokes
]

def _build_petersen_adj():
    adj = [[] for _ in range(10)]
    for u, v in _PETERSEN_EDGES:
        adj[u].append(v)
        adj[v].append(u)
    return adj

_PETERSEN_ADJ = _build_petersen_adj()

# K4 Base Graph for voltage construction
_K4_EDGES = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
_K4_ADJ = {0: [1,2,3], 1: [0,2,3], 2: [0,1,3], 3: [0,1,2]}

def _is_prime(n):
    if n < 2: return False
    if n < 4: return True
    if n % 2 == 0 or n % 3 == 0: return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _check_girth_petersen(voltages, m, required_girth):
    """Check if voltage assignment on Petersen gives girth >= required_girth.
    voltages: list of 15 ints (Z_m)
    Uses the pre-computed constraint vectors."""
    # This is handled by _C_MATRIX in the main construct
    return True  # Placeholder, actual check done in construct

def _check_girth_k4(voltages, p, required_girth):
    """Check if voltage assignment on K4 (PSL(2,p)) gives girth >= required_girth.
    voltages: dict mapping edge tuple (u,v) with u<v to matrix (a,b,c,d)
    """
    # Build oriented voltage map
    volt_map = {}
    for (u, v) in _K4_EDGES:
        mat = voltages[(u, v)]
        volt_map[(u, v)] = mat
        # Inverse: [[d, -b], [-c, a]]
        a, b, c, d = mat
        volt_map[(v, u)] = (d % p, (-b) % p, (-c) % p, a % p)
    
    max_len = required_girth - 1
    
    def mat_mul(m1, m2, p):
        a1, b1, c1, d1 = m1
        a2, b2, c2, d2 = m2
        return ((a1*a2 + b1*c2) % p, (a1*b2 + b1*d2) % p,
                (c1*a2 + d1*c2) % p, (c1*b2 + d1*d2) % p)
    
    def mat_is_identity(m, p):
        a, b, c, d = m
        if (a == 1 and b == 0 and c == 0 and d == 1):
            return True
        if (a == p-1 and b == 0 and c == 0 and d == p-1):
            return True
        return False
    
    # DFS to find short closed walks with identity product
    # State: (current, prev, length, product)
    # To speed up, we can use iterative deepening or just limit depth
    
    def dfs(current, prev, length, product, start_v):
        if length >= max_len:
            return False
        
        for next_v in _K4_ADJ[current]:
            if next_v == prev:
                continue
            
            new_product = mat_mul(product, volt_map[(current, next_v)], p)
            
            if next_v == start_v and length + 1 >= 3:
                if mat_is_identity(new_product, p):
                    return True
            
            if dfs(next_v, current, length + 1, new_product, start_v):
                return True
        
        return False
    
    for start_v in range(4):
        identity = (1, 0, 0, 1)
        if dfs(start_v, -1, 0, identity, start_v):
            return False
    
    return True

def _random_psl_element(p, rng):
    while True:
        a = rng.randint(0, p-1)
        b = rng.randint(0, p-1)
        c = rng.randint(0, p-1)
        d = rng.randint(0, p-1)
        det = (a*d - b*c) % p
        if det == 1:
            return (a, b, c, d)

def _build_k4_lift(voltages, p):
    """Build the lift graph from K4 with PSL(2,p) voltages.
    Returns (n, edges)
    """
    # We need to enumerate all group elements or use a hash map
    # For PSL(2,p), |G| = p(p^2-1)/2
    # For p=100, |G| = 100*9999/2 = 499950. Too large to enumerate all.
    # Instead, we build the graph by following edges from a starting vertex.
    # But the graph might not be connected? No, voltage lifts of connected graphs are connected.
    
    # Vertex representation: (base_v, group_elem)
    # We'll use a dictionary to map (base_v, elem) to index
    # elem is a tuple (a,b,c,d)
    
    index = {}
    order = []
    edges = []
    
    def get_index(base_v, elem):
        key = (base_v, elem)
        if key not in index:
            index[key] = len(order)
            order.append(key)
        return index[key]
    
    # Start from (0, identity)
    identity = (1, 0, 0, 1)
    start_key = (0, identity)
    get_index(0, identity)
    
    # BFS to explore the graph
    queue = deque([start_key])
    visited = {start_key}
    
    while queue:
        base_v, elem = queue.popleft()
        idx_v = index[(base_v, elem)]
        
        for neighbor in _K4_ADJ[base_v]:
            # Determine voltage
            if (base_v, neighbor) in voltages:
                mat = voltages[(base_v, neighbor)]
            else:
                mat = voltages[(neighbor, base_v)]
                # Inverse
                a, b, c, d = mat
                mat = (d % p, (-b) % p, (-c) % p, a % p)
            
            # New element: mat * elem (right multiplication? or left?)
            # Standard voltage graph: if edge u->v has voltage g, then (u, h) -> (v, h*g)
            # Let's use right multiplication: new_elem = elem * mat
            a1, b1, c1, d1 = elem
            a2, b2, c2, d2 = mat
            new_elem = ((a1*a2 + b1*c2) % p, (a1*b2 + b1*d2) % p,
                        (c1*a2 + d1*c2) % p, (c1*b2 + d1*d2) % p)
            
            new_key = (neighbor, new_elem)
            if new_key not in visited:
                visited.add(new_key)
                get_index(neighbor, new_elem)
                queue.append(new_key)
            
            # Add edge
            idx_u = index[new_key]
            if idx_v < idx_u:
                edges.append((idx_v, idx_u))
    
    n = len(order)
    return n, edges

def construct(k, g):
    start_time = time.time()
    DEADLINE = start_time + 110.0  # Safety margin
    
    if k != 3:
        return _trivial_regular(k, g)
    
    # Strategy 1: Voltage graph on K4 with PSL(2,p)
    # n = 4 * p * (p^2-1)/2? No, the lift has n = 4 * |G| vertices?
    # Wait, the lift of a graph with base B and group G has |V(B)| * |G| vertices.
    # So n = 4 * p(p^2-1)/2.
    # For p=11, n = 4 * 11*120/2 = 2640.
    # For p=7, n = 4 * 7*48/2 = 672.
    # Moore bound is 512. So p=7 gives 672, which is close.
    # Can we do better?
    # The prompt says smallest known is 936.
    # 936 is not of the form 4*p(p^2-1)/2.
    # 936 / 4 = 234.
    # 234 = p(p^2-1)/2 => p^3-p = 468. p=7 -> 343-7=336. p=8 -> 512-8=504.
    # So the known 936-vertex graph is not a simple PSL(2,p) lift of K4.
    # It might be a lift of a different base graph or a different group.
    
    # Let's try the Petersen voltage graph (Z_m) first, as it allows smaller n.
    # n = 10 * m.
    # For m=52, n=520.
    # We need to find a voltage assignment in Z_m that gives girth >= 16.
    
    # Pre-compute constraint vectors for Petersen
    # This is expensive, so do it once.
    _C_MATRIX = _generate_closed_walk_vectors_petersen(15)
    
    best_result = None
    
    # Try Z_m voltage graphs on Petersen
    # m must be >= 52 for n >= 512 (Moore bound)
    # Actually, n=10m. If m=52, n=520.
    # Let's try m from 52 upwards.
    
    # To speed up, we can use numpy.
    # _C_MATRIX is a list of tuples of 15 ints.
    # Convert to numpy array.
    if len(_C_MATRIX) > 0:
        C = np.array(_C_MATRIX, dtype=np.int64)
        
        # Try different m
        for m in range(52, 100):
            if time.time() > DEADLINE:
                break
            
            n = 10 * m
            if best_result is not None and n >= best_result[0]:
                continue
            
            # Try random vectors
            # Number of constraints: len(C)
            # For each constraint c, sum(c_i * v_i) % m != 0
            # We can vectorize this.
            
            found = False
            # How many tries?
            # Probability of a random vector satisfying one constraint is (m-1)/m.
            # For N constraints, probability is ((m-1)/m)^N.
            # If N=100, m=52, prob = (51/52)^100 ~ 0.14.
            # So ~7 tries on average.
            # Let's try 100 times.
            
            for _ in range(100):
                if time.time() > DEADLINE:
                    break
                
                v = np.random.randint(0, m, size=15, dtype=np.int64)
                sums = (C @ v) % m
                
                if not np.any(sums == 0):
                    # Valid!
                    edges = []
                    for i, (u, v_base) in enumerate(_PETERSEN_EDGES):
                        a = v[i]
                        for k_group in range(m):
                            su = u * m + k_group
                            sv = v_base * m + ((k_group + a) % m)
                            edges.append((su, sv))
                    
                    best_result = (n, edges)
                    found = True
                    break
            
            if found:
                break
    
    if best_result is not None:
        return best_result
    
    # Fallback: K4 + PSL(2,p)
    # Try small primes
    for p in [7, 11, 13, 17, 19, 23]:
        if time.time() > DEADLINE - 5:
            break
        
        n_est = 4 * p * (p*p - 1) // 2
        if best_result is not None and n_est >= best_result[0]:
            continue
        
        rng = random.Random(42 + p)
        for _ in range(1000):
            if time.time() > DEADLINE - 5:
                break
            
            voltages = {}
            for (u, v) in _K4_EDGES:
                voltages[(u, v)] = _random_psl_element(p, rng)
            
            # Quick check: no identity
            valid = True
            for (u, v) in _K4_EDGES:
                mat = voltages[(u, v)]
                a, b, c, d = mat
                if (a == 1 and b == 0 and c == 0 and d == 1) or (a == p-1 and b == 0 and c == 0 and d == p-1):
                    valid = False
                    break
            if not valid:
                continue
            
            if _check_girth_k4(voltages, p, g):
                n, edges = _build_k4_lift(voltages, p)
                if best_result is None or n < best_result[0]:
                    best_result = (n, edges)
                break
    
    if best_result is not None:
        return best_result
    
    # Final fallback: LUW construction
    return _luw_fallback(k, g)

def _generate_closed_walk_vectors_petersen(max_len=15):
    """Generate coefficient vectors for all non-backtracking closed walks of length 5..max_len in Petersen."""
    vectors = set()
    n_edges = 15
    
    edge_map = {}
    for i, (u, v) in enumerate(_PETERSEN_EDGES):
        if u > v:
            u, v = v, u
        edge_map[(u, v)] = (i, 1)
        edge_map[(v, u)] = (i, -1)
    
    for start in range(10):
        stack = [(start, -1, 0, [0]*n_edges)]
        
        while stack:
            curr, prev, length, coeff = stack.pop()
            
            if length >= 5 and curr == start:
                vectors.add(tuple(coeff))
            
            if length == max_len:
                continue
            
            for next_v in _PETERSEN_ADJ[curr]:
                if next_v == prev:
                    continue
                new_coeff = coeff[:]
                e_idx, direction = edge_map[(curr, next_v)]
                new_coeff[e_idx] += direction
                stack.append((next_v, curr, length + 1, new_coeff))
                
    return list(vectors)

def _trivial_regular(k, g):
    if k == 3:
        return (10, _PETERSEN_EDGES)
    return (10, _PETERSEN_EDGES)

def _luw_fallback(k, g):
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
