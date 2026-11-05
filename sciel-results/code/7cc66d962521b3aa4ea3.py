import time
import random
import sys
from collections import deque
import numpy as np

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Strategy:
      1. Try voltage lift of Petersen graph with linear algebra over F_p
      2. Fallback to Heawood voltage lift
      3. Final fallback: LUW construction
    """
    start_time = time.time()
    deadline = start_time + 135.0

    if k == 3:
        # Try Petersen voltage lift first (best known approach)
        n_petersen, edges_petersen = _construct_petersen_lift(g, deadline, start_time)
        if n_petersen is not None:
            return n_petersen, edges_petersen
        
        # Try Heawood voltage lift
        n_heawood, edges_heawood = _construct_heawood_lift(g, deadline, start_time)
        if n_heawood is not None:
            return n_heawood, edges_heawood

        # Final fallback
        return _construct_luw(k, g, d=11)
    else:
        return _construct_luw(k, g, d=7)


def _get_petersen_edges():
    return [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]

def _get_heawood_edges():
    # Fano plane incidence graph
    # Points 0-6, Lines 7-13
    edges = []
    # L0: 0,1,2 -> 7
    edges += [(0,7), (1,7), (2,7)]
    # L1: 0,3,4 -> 8
    edges += [(0,8), (3,8), (4,8)]
    # L2: 0,5,6 -> 9
    edges += [(0,9), (5,9), (6,9)]
    # L3: 1,3,5 -> 10
    edges += [(1,10), (3,10), (5,10)]
    # L4: 1,4,6 -> 11
    edges += [(1,11), (4,11), (6,11)]
    # L5: 2,3,6 -> 12
    edges += [(2,12), (3,12), (6,12)]
    # L6: 2,4,5 -> 13
    edges += [(2,13), (4,13), (5,13)]
    return edges


def _construct_petersen_lift(g, deadline, start_time):
    """
    Constructs a girth-g graph by lifting the Petersen graph with voltages in Z_m.
    Uses linear algebra over F_p to find valid voltage assignments.
    """
    base_edges = _get_petersen_edges()
    n_base = 10
    m_edges = len(base_edges)
    
    # Precompute all non-backtracking closed walks of length < g in the base graph
    # Each closed walk gives a linear constraint on voltages
    coeff_vectors = set()
    max_walk_len = g - 1
    
    # Build adjacency for base graph
    adj = [[] for _ in range(n_base)]
    edge_index = {}
    for i, (u, v) in enumerate(base_edges):
        adj[u].append((v, i))
        adj[v].append((u, i))
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
    
    # DFS to find all non-backtracking closed walks of length < g
    for start_v in range(n_base):
        # State: (current, prev, length, counts_vector)
        # counts_vector[e] = net number of times edge e is traversed (signed)
        stack = [(start_v, -1, 0, [0] * m_edges)]
        
        while stack:
            v, prev_v, l, counts = stack.pop()
            
            if l >= max_walk_len:
                continue
            
            for next_v, e_idx in adj[v]:
                if next_v == prev_v:
                    continue
                
                new_counts = counts.copy()
                u_edge, v_edge = base_edges[e_idx]
                if v == u_edge and next_v == v_edge:
                    new_counts[e_idx] += 1
                else:
                    new_counts[e_idx] -= 1
                
                new_l = l + 1
                if next_v == start_v and new_l > 0:
                    # Found a closed walk
                    coeff_vectors.add(tuple(new_counts))
                
                stack.append((next_v, v, new_l, new_counts))
    
    # Normalize constraints
    unique_coeffs = set()
    for c in coeff_vectors:
        if all(x == 0 for x in c):
            continue
        # Normalize: make first non-zero entry positive
        for i in range(len(c)):
            if c[i] != 0:
                if c[i] < 0:
                    c = tuple(-x for x in c)
                unique_coeffs.add(c)
                break
    coeff_list = list(unique_coeffs)
    
    if not coeff_list:
        # No constraints, any voltages work
        voltages = [1] * m_edges
        m = 2
        n_candidate = n_base * m
        edges = _build_lifted_graph(base_edges, voltages, m)
        return n_candidate, edges
    
    # Build coefficient matrix
    coeff_matrix = np.array(coeff_list, dtype=np.int64)
    n_constraints = len(coeff_matrix)
    
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

    # Try small primes to minimize n = n_base * m
    primes = []
    candidate = 2
    while len(primes) < 300:
        if is_prime(candidate):
            primes.append(candidate)
        candidate += 1
    
    best_n = float('inf')
    best_edges = None
    
    random.seed(42)
    
    for m in primes:
        if time.time() > deadline - 10:
            break
        
        n_candidate = n_base * m
        if n_candidate >= best_n:
            continue
        
        # Reduce constraints modulo m
        coeff_mod = coeff_matrix % m
        
        # Remove zero rows
        nonzero_mask = np.any(coeff_mod != 0, axis=1)
        coeff_mod = coeff_mod[nonzero_mask]
        
        if len(coeff_mod) == 0:
            # No constraints mod m, any voltages work
            voltages = np.random.randint(1, m, size=m_edges).astype(np.int64)
            edges = _build_lifted_graph(base_edges, voltages.tolist(), m)
            return n_candidate, edges
        
        # Use Gaussian elimination to find a solution
        # We want voltages such that coeff_matrix @ voltages ≡ 0 (mod m)
        # and no voltage is 0 (to avoid trivial cycles)
        
        # First, find the null space dimension
        A = coeff_mod.copy()
        n_rows = len(A)
        n_cols = m_edges
        
        # Gaussian elimination over F_p
        rank = 0
        pivots = []
        for col in range(n_cols):
            # Find pivot row
            pivot_row = -1
            for row in range(rank, n_rows):
                if A[row, col] != 0:
                    pivot_row = row
                    break
            
            if pivot_row == -1:
                continue
            
            # Swap rows
            A[[rank, pivot_row]] = A[[pivot_row, rank]]
            
            # Normalize pivot row
            inv_pivot = pow(A[rank, col], m - 2, m)
            A[rank, :] = (A[rank, :] * inv_pivot) % m
            
            # Eliminate other rows
            for row in range(n_rows):
                if row != rank and A[row, col] != 0:
                    factor = A[row, col]
                    A[row, :] = (A[row, :] - factor * A[rank, :]) % m
            
            pivots.append(col)
            rank += 1
        
        free_cols = [c for c in range(n_cols) if c not in pivots]
        nullity = len(free_cols)
        
        if nullity == 0:
            # Only trivial solution, skip
            continue
        
        # Generate random solutions in null space
        # Set free variables to random values, solve for pivot variables
        found = False
        max_attempts = 200
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Random assignment to free variables
            voltages = np.zeros(m_edges, dtype=np.int64)
            for fc in free_cols:
                voltages[fc] = random.randint(1, m - 1)
            
            # Solve for pivot variables using back substitution
            for i in range(rank - 1, -1, -1):
                pivot_col = pivots[i]
                val = A[i, pivot_col]  # Should be 1 after normalization
                # Sum of known terms
                known_sum = 0
                for j in range(pivot_col + 1, m_edges):
                    if A[i, j] != 0:
                        known_sum = (known_sum + A[i, j] * voltages[j]) % m
                
                voltages[pivot_col] = (-known_sum) % m
            
            # Check if any voltage is 0
            if np.any(voltages == 0):
                continue
            
            # Verify the solution
            products = (coeff_matrix @ voltages) % m
            if np.all(products == 0):
                # Valid solution found
                edges = _build_lifted_graph(base_edges, voltages.tolist(), m)
                return n_candidate, edges
        
        # If we haven't found a solution for this m, try next m
    
    return None, None


def _construct_heawood_lift(g, deadline, start_time):
    """
    Constructs a girth-g graph by lifting the Heawood graph with voltages in Z_m.
    """
    base_edges = _get_heawood_edges()
    n_base = 14
    m_edges = len(base_edges)
    
    # Precompute all non-backtracking closed walks of length < g in the base graph
    coeff_vectors = set()
    max_walk_len = g - 1
    
    # Build adjacency for base graph
    adj = [[] for _ in range(n_base)]
    edge_index = {}
    for i, (u, v) in enumerate(base_edges):
        adj[u].append((v, i))
        adj[v].append((u, i))
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
    
    # DFS to find all non-backtracking closed walks of length < g
    for start_v in range(n_base):
        stack = [(start_v, -1, 0, [0] * m_edges)]
        
        while stack:
            v, prev_v, l, counts = stack.pop()
            
            if l >= max_walk_len:
                continue
            
            for next_v, e_idx in adj[v]:
                if next_v == prev_v:
                    continue
                
                new_counts = counts.copy()
                u_edge, v_edge = base_edges[e_idx]
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
        for i in range(len(c)):
            if c[i] != 0:
                if c[i] < 0:
                    c = tuple(-x for x in c)
                unique_coeffs.add(c)
                break
    coeff_list = list(unique_coeffs)
    
    if not coeff_list:
        voltages = [1] * m_edges
        m = 2
        n_candidate = n_base * m
        edges = _build_lifted_graph(base_edges, voltages, m)
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
    while len(primes) < 300:
        if is_prime(candidate):
            primes.append(candidate)
        candidate += 1
    
    best_n = float('inf')
    best_edges = None
    
    random.seed(123)
    
    for m in primes:
        if time.time() > deadline - 10:
            break
        
        n_candidate = n_base * m
        if n_candidate >= best_n:
            continue
        
        coeff_mod = coeff_matrix % m
        nonzero_mask = np.any(coeff_mod != 0, axis=1)
        coeff_mod = coeff_mod[nonzero_mask]
        
        if len(coeff_mod) == 0:
            voltages = np.random.randint(1, m, size=m_edges).astype(np.int64)
            edges = _build_lifted_graph(base_edges, voltages.tolist(), m)
            return n_candidate, edges
        
        A = coeff_mod.copy()
        n_rows = len(A)
        n_cols = m_edges
        
        rank = 0
        pivots = []
        for col in range(n_cols):
            pivot_row = -1
            for row in range(rank, n_rows):
                if A[row, col] != 0:
                    pivot_row = row
                    break
            
            if pivot_row == -1:
                continue
            
            A[[rank, pivot_row]] = A[[pivot_row, rank]]
            inv_pivot = pow(A[rank, col], m - 2, m)
            A[rank, :] = (A[rank, :] * inv_pivot) % m
            
            for row in range(n_rows):
                if row != rank and A[row, col] != 0:
                    factor = A[row, col]
                    A[row, :] = (A[row, :] - factor * A[rank, :]) % m
            
            pivots.append(col)
            rank += 1
        
        free_cols = [c for c in range(n_cols) if c not in pivots]
        nullity = len(free_cols)
        
        if nullity == 0:
            continue
        
        max_attempts = 200
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            voltages = np.zeros(m_edges, dtype=np.int64)
            for fc in free_cols:
                voltages[fc] = random.randint(1, m - 1)
            
            for i in range(rank - 1, -1, -1):
                pivot_col = pivots[i]
                known_sum = 0
                for j in range(pivot_col + 1, m_edges):
                    if A[i, j] != 0:
                        known_sum = (known_sum + A[i, j] * voltages[j]) % m
                
                voltages[pivot_col] = (-known_sum) % m
            
            if np.any(voltages == 0):
                continue
            
            products = (coeff_matrix @ voltages) % m
            if np.all(products == 0):
                edges = _build_lifted_graph(base_edges, voltages.tolist(), m)
                return n_candidate, edges
    
    return None, None


def _build_lifted_graph(base_edges, voltages, m):
    n_base = len(set(v for e in base_edges for v in e))
    edge_index = {}
    for i, (u, v) in enumerate(base_edges):
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
            u_edge, v_edge = base_edges[e_idx]
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
    """
    LUW construction with specified depth d and variant for rule selection.
    For k=3, uses q=3 arithmetic.
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
