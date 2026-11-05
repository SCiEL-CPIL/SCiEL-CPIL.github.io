import time
import random
import sys
from collections import deque
import numpy as np

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Uses voltage lift of Petersen graph for k=3 with Simulated Annealing.
    """
    start_time = time.time()
    deadline = start_time + 140.0

    if k == 3:
        return _construct_petersen_lift(g, deadline, start_time)
    else:
        return _construct_luw(k, g, d=7)

def _construct_petersen_lift(g, deadline, start_time):
    """
    Constructs a girth-g graph by lifting the Petersen graph with voltages in Z_m.
    Uses Simulated Annealing to find voltages.
    """
    # Petersen graph edges (10 vertices, 15 edges)
    pet_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),  # Outer
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),  # Inner
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)   # Spokes
    ]
    
    n_base = 10
    edge_index = {}
    for i, (u, v) in enumerate(pet_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
    
    adj = [[] for _ in range(n_base)]
    for u, v in pet_edges:
        adj[u].append(v)
        adj[v].append(u)

    # Precompute all non-backtracking closed walks of length < g in the Petersen graph
    coeff_vectors = set()
    max_walk_len = g - 1
    
    # We only care about walks of length >= 3 (since k=3, no 2-cycles, no 1-cycles)
    # Actually, non-backtracking walks of length 1 and 2 don't close in a simple graph without loops/multi-edges.
    # Petersen is simple.
    
    for start_v in range(n_base):
        stack = [(start_v, -1, 0, [0] * 15)]
        
        while stack:
            v, prev_v, l, counts = stack.pop()
            
            if l >= max_walk_len:
                continue
            
            for next_v in adj[v]:
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
                    # Normalize to avoid duplicates (reverse walk)
                    c = new_counts
                    # Find first non-zero
                    for i in range(15):
                        if c[i] != 0:
                            if c[i] < 0:
                                c = tuple(-x for x in c)
                            else:
                                c = tuple(c)
                            break
                    if any(x != 0 for x in c):
                        coeff_vectors.add(c)
                
                stack.append((next_v, v, new_l, new_counts))
    
    coeff_list = list(coeff_vectors)
    
    if len(coeff_list) == 0:
        # Fallback, shouldn't happen for g >= 3
        return n_base, pet_edges

    # Build numpy matrix for vectorized evaluation
    coeff_matrix = np.array(coeff_list, dtype=np.int64)
    n_constraints = len(coeff_matrix)
    
    # Calculate Moore bound to estimate minimum m
    # For k=3, g=16 (even), Moore bound is 2 * (2^(g/2) - 1)
    if g % 2 == 0:
        moore_bound = 2 * ((2 ** (g // 2)) - 1)
    else:
        moore_bound = 2 * (2 ** ((g - 1) // 2))
    
    m_min = max(2, (moore_bound + n_base - 1) // n_base)
    
    # Generate candidate moduli
    # We want small n = 10 * m.
    # Try primes and some composites.
    candidates_m = []
    # Add m_min and a range around it
    current_m = m_min
    while len(candidates_m) < 50 and current_m < 500:
        candidates_m.append(current_m)
        current_m += 1
    
    # Ensure we have some primes in there if m_min is small, but composites are fine too.
    # Actually, prime moduli make the "non-zero" constraint easier to satisfy locally?
    # If m is prime, c_j is invertible.
    # Let's stick to a mix. The list above is just consecutive integers.
    
    best_n = float('inf')
    best_edges = None
    
    for m in candidates_m:
        if time.time() > deadline:
            break
            
        n_candidate = 10 * m
        if n_candidate >= best_n:
            continue
        
        # Try to solve for this m
        solution = _solve_voltage_lift_sa(coeff_matrix, m, deadline, start_time)
        
        if solution is not None:
            edges = _build_lifted_graph(pet_edges, solution, m)
            # Optional: Verify girth (expensive, skip if confident)
            return n_candidate, edges
    
    # Fallback if no solution found in candidates
    return _construct_luw(k=3, g=g, d=11)

def _solve_voltage_lift_sa(C, m, deadline, start_time):
    """
    Simulated Annealing to find x in Z_m^15 such that C @ x != 0 mod m for all rows.
    """
    n_edges = 15
    n_constraints = len(C)
    
    if n_constraints == 0:
        return [1] * n_edges
        
    # Precompute non-zero indices for each constraint to speed up violation checking
    # Actually, we just compute C @ x and check for zeros.
    # C @ x is vectorized.
    
    # Initial solution: random
    x = np.array([random.randint(0, m - 1) for _ in range(n_edges)], dtype=np.int64)
    
    def get_energy(x):
        # Returns number of rows where (C @ x) % m == 0
        sums = (C @ x) % m
        return np.sum(sums == 0)
    
    current_energy = get_energy(x)
    if current_energy == 0:
        return x.tolist()
        
    best_x = x.copy()
    best_energy = current_energy
    
    # SA Parameters
    # Time budget per m: ~5 seconds?
    # Total time 140s. 50 candidates -> 2.8s each.
    # Let's use 3s per candidate.
    time_limit = start_time + 3.0
    
    T = 1.0
    cooling_rate = 0.99
    min_T = 0.01
    
    while time.time() < time_limit and T > min_T:
        # Pick a random edge to change
        j = random.randint(0, n_edges - 1)
        old_val = x[j]
        
        # Pick a new random value
        new_val = random.randint(0, m - 1)
        if new_val == old_val:
            continue
            
        x[j] = new_val
        
        # Calculate delta energy
        # Only rows where C[row, j] != 0 are affected
        # But calculating full C @ x is fast enough for 15 vars.
        new_energy = get_energy(x)
        delta_E = new_energy - current_energy
        
        if delta_E <= 0 or random.random() < np.exp(-delta_E / T):
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_x = x.copy()
                if best_energy == 0:
                    return best_x.tolist()
        else:
            x[j] = old_val
            
        T *= cooling_rate
        
    return None if best_energy > 0 else best_x.tolist()

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
