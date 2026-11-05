import time
import random
import sys
from collections import deque
import numpy as np

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    For k=3, uses voltage-lifted Petersen graph with backtracking CSP.
    Falls back to LUW construction if needed.
    """
    start_time = time.time()
    deadline = start_time + 140.0

    if k == 3 and g >= 8:
        result = _petersen_voltage_lift(g, deadline, start_time)
        if result is not None:
            return result
    
    return _construct_luw(k, g, d=7)

def _petersen_voltage_lift(g, deadline, start_time):
    """
    Find small 3-regular graphs with girth >= g via voltage lifting of Petersen graph.
    Uses backtracking CSP on non-backtracking closed walk constraints.
    """
    pet_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # Outer cycle
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),  # Inner cycle
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)   # Spokes
    ]
    
    n_base = 10
    edge_index = {}
    for i, (u, v) in enumerate(pet_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
    
    # Precompute all non-backtracking closed walks of length in [4, g-1]
    # Each constraint: (coeff_vector, length) where sum(coeff[i] * voltage[i]) = 0 mod m means a bad cycle
    constraints = []
    max_walk_len = g - 1
    min_walk_len = 4  # smallest cycle length in 3-regular graph
    
    for start_v in range(n_base):
        # DFS with state: (current_vertex, previous_vertex, length, coeff_list)
        # To keep memory manageable, use iterative deepening
        stack = [(start_v, -1, 0, [0] * 15)]
        
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
                
                if next_v == start_v and new_l >= min_walk_len:
                    # This is a closed walk; add as constraint
                    c = tuple(new_counts)
                    # Normalize: make first non-zero entry positive
                    found_nonzero = False
                    for i in range(15):
                        if c[i] != 0:
                            if c[i] < 0:
                                c = tuple(-x for x in c)
                            found_nonzero = True
                            break
                    if found_nonzero and c not in [x[0] for x in constraints]:
                        constraints.append((c, new_l))
                
                if new_l < max_walk_len:
                    stack.append((next_v, v, new_l, new_counts))
    
    # Deduplicate constraints
    unique_constraints = {}
    for c, l in constraints:
        if c not in unique_constraints or unique_constraints[c] > l:
            unique_constraints[c] = l
    
    # Sort by length (shorter cycles are more constraining) and then by density
    sorted_constraints = sorted(unique_constraints.items(), key=lambda x: (x[1], -sum(1 for v in x[0] if v != 0)))
    sorted_constraints = [(c, l) for c, l in sorted_constraints]
    
    if not sorted_constraints:
        return None
    
    n_constraints = len(sorted_constraints)
    
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
    
    # Try primes starting from small values
    # For g=16, we need n = 10*m, so m >= 40 gives n >= 400
    # Try primes near the expected range
    candidate_primes = []
    # Start from around 40 and go up
    for p in range(37, 200):
        if is_prime(p):
            candidate_primes.append(p)
    
    # Also add some larger primes for safety
    for p in [211, 223, 227, 229, 233, 239]:
        candidate_primes.append(p)
    
    candidate_primes = sorted(set(candidate_primes))
    
    best_n = float('inf')
    best_edges = None
    
    for m in candidate_primes:
        if time.time() > deadline:
            break
        
        n_candidate = 10 * m
        if n_candidate >= best_n:
            break
        
        # Backtracking CSP
        voltages = [0] * 15
        domain = list(range(1, m))  # Non-zero elements mod m
        
        # Use random seed for reproducibility within this call
        random.seed(42)
        
        def backtrack():
            if time.time() > deadline:
                return False
            
            # Find next unassigned variable
            var = -1
            # MRV heuristic: pick variable with smallest domain given current assignments
            best_domain = m
            for i in range(15):
                if voltages[i] == 0 and i != -1:
                    # Count how many values are still valid
                    # For simplicity, just use order
                    var = i
                    break
            
            if var == -1:
                # All assigned, check all constraints
                for c, l in sorted_constraints:
                    s = 0
                    for i in range(15):
                        if c[i] != 0:
                            s += c[i] * voltages[i]
                    if s % m == 0:
                        return False
                return True
            
            # Try values in random order
            values = domain[:]
            random.shuffle(values)
            
            for val in values:
                voltages[var] = val
                
                # Check partial constraints
                valid = True
                for c, l in sorted_constraints:
                    s = 0
                    all_assigned = True
                    for i in range(15):
                        if c[i] != 0:
                            if voltages[i] == 0:
                                all_assigned = False
                                break
                            s += c[i] * voltages[i]
                    if all_assigned:
                        if s % m == 0:
                            valid = False
                            break
                    elif s % m == 0:
                        # If partial sum is already 0 mod m, and all remaining
                        # coefficients are 0, we'd have a problem, but since
                        # some are unassigned (which must be non-zero), we can
                        # still potentially satisfy
                        pass
                
                if not valid:
                    voltages[var] = 0
                    continue
                
                if backtrack():
                    return True
                else:
                    voltages[var] = 0
            
            return False
        
        # Multiple attempts with different random orderings
        found = False
        num_attempts = min(50, max(5, 300 // (n_candidate // 10)))
        
        for attempt in range(num_attempts):
            if time.time() > deadline:
                break
            
            voltages = [0] * 15
            # Different seed for each attempt
            random.seed(attempt * 1000 + m)
            
            if backtrack():
                # Valid voltage assignment found
                edges = _build_lifted_graph(pet_edges, voltages, m)
                return n_candidate, edges
        
    return None

def _build_lifted_graph(pet_edges, voltages, m):
    """Build the lifted graph from voltages."""
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
