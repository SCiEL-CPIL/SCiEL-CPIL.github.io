import time
import random
import sys
from collections import deque
import numpy as np

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Uses voltage lift of Petersen graph for k=3 with backtracking constraint solver.
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
    Uses backtracking constraint solver with forward checking.
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
    
    # Precompute all non-backtracking closed walks of length < g in the Petersen graph
    coeff_vectors = set()
    max_walk_len = g - 1
    
    for start_v in range(n_base):
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
    
    # Build numpy matrix for vectorized evaluation
    if len(coeff_list) > 0:
        coeff_matrix = np.array(coeff_list, dtype=np.int64)
        n_constraints = len(coeff_matrix)
    else:
        coeff_matrix = np.zeros((0, 15), dtype=np.int64)
        n_constraints = 0
    
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

    # Generate candidate moduli: primes first, then some composites
    candidates_m = []
    candidate = 2
    while len(candidates_m) < 100:
        if is_prime(candidate):
            candidates_m.append(candidate)
        candidate += 1
    
    best_n = float('inf')
    best_edges = None
    
    for m in candidates_m:
        if time.time() > deadline:
            break
        
        n_candidate = 10 * m
        if n_candidate >= best_n:
            continue
        
        # Try backtracking constraint solver
        solution = _backtrack_solve(coeff_list, n_constraints, m, deadline, start_time)
        if solution is not None:
            edges = _build_lifted_graph(pet_edges, solution, m)
            return n_candidate, edges
        
        # If backtracking fails, try random+repair as fallback
        max_attempts = 200
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            voltages = np.array([random.randint(1, m - 1) for _ in range(15)], dtype=np.int64)
            
            for repair_step in range(30):
                if n_constraints > 0:
                    sums = (coeff_matrix @ voltages) % m
                    violated_indices = np.where(sums == 0)[0]
                    
                    if len(violated_indices) == 0:
                        edges = _build_lifted_graph(pet_edges, voltages.tolist(), m)
                        return n_candidate, edges
                    
                    viol_idx = random.choice(violated_indices.tolist())
                    c = coeff_list[viol_idx]
                    candidates = [i for i in range(15) if c[i] != 0]
                    if not candidates:
                        break
                    j = random.choice(candidates)
                    delta = random.randint(1, m - 1)
                    voltages[j] = (voltages[j] + delta) % m
                    if voltages[j] == 0:
                        voltages[j] = 1
                else:
                    edges = _build_lifted_graph(pet_edges, voltages.tolist(), m)
                    return n_candidate, edges
    
    # Fallback
    return _construct_luw(k=3, g=g, d=11)

def _backtrack_solve(coeff_list, n_constraints, m, deadline, start_time):
    """
    Backtracking solver for voltage assignment with forward checking.
    Assigns voltages to 15 edges one at a time, checking constraints.
    """
    if n_constraints == 0:
        return [1] * 15
    
    # Precompute for each edge, which constraints it participates in
    edge_to_constraints = [[] for _ in range(15)]
    for ci, c in enumerate(coeff_list):
        for ei in range(15):
            if c[ei] != 0:
                edge_to_constraints[ei].append(ci)
    
    # Order edges by number of constraints they affect (most constrained first)
    edge_order = sorted(range(15), key=lambda e: -len(edge_to_constraints[e]))
    
    # Precompute constraint coefficients per edge for faster checking
    # For each constraint, store (edge_idx, coeff) pairs
    constraint_terms = []
    for ci, c in enumerate(coeff_list):
        terms = [(ei, c[ei]) for ei in range(15) if c[ei] != 0]
        constraint_terms.append(terms)
    
    def backtrack(assignment, edge_pos, time_limit):
        """
        assignment: dict mapping edge_idx -> voltage
        edge_pos: index into edge_order
        Returns solution list or None
        """
        if time.time() > time_limit:
            return None
        
        if edge_pos == 15:
            # Check all constraints
            for ci, terms in enumerate(constraint_terms):
                s = 0
                for ei, coeff in terms:
                    s += coeff * assignment[ei]
                if s % m == 0:
                    return None
            return [assignment[ei] for ei in range(15)]
        
        e = edge_order[edge_pos]
        
        # Try each possible voltage value
        values = list(range(1, m))
        random.shuffle(values)
        
        for val in values:
            if time.time() > time_limit:
                return None
            
            assignment[e] = val
            
            # Forward checking: see if any constraint is now determined and violated
            # A constraint is determined if all its edges are assigned
            feasible = True
            for ci, terms in enumerate(constraint_terms):
                # Check if all edges in this constraint are assigned
                all_assigned = all(assignment.get(ei) is not None for ei, coeff in terms)
                if all_assigned:
                    s = sum(coeff * assignment[ei] for ei, coeff in terms)
                    if s % m == 0:
                        feasible = False
                        break
                else:
                    # Check if constraint can still be satisfied
                    # Current partial sum + remaining must be != 0 mod m
                    partial = 0
                    remaining_edges = []
                    for ei, coeff in terms:
                        if ei in assignment:
                            partial += coeff * assignment[ei]
                        else:
                            remaining_edges.append((ei, coeff))
                    
                    if remaining_edges:
                        # For prime m, as long as at least one remaining edge has nonzero coeff,
                        # we can always choose a value to make the sum nonzero
                        # So no pruning needed here for prime m
                        pass
                    else:
                        if partial % m == 0:
                            feasible = False
                            break
            
            if feasible:
                result = backtrack(assignment, edge_pos + 1, time_limit)
                if result is not None:
                    return result
            
            del assignment[e]
        
        return None
    
    time_limit = min(deadline, start_time + 10.0)  # Limit backtracking to 10s
    assignment = {}
    result = backtrack(assignment, 0, time_limit)
    return result

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
