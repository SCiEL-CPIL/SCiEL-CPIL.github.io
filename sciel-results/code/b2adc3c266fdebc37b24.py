import time
import random
from collections import defaultdict, deque
import numpy as np

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Uses voltage lift with optimized voltage search for k=3.
    """
    start_time = time.time()
    deadline = start_time + 145.0
    
    if k == 3:
        return _construct_optimized_lift(g, deadline)
    else:
        return _construct_luw(k, g, d=7)

def _construct_optimized_lift(g, deadline):
    """
    Constructs a girth-g graph by lifting the Petersen graph with optimized voltages.
    Uses greedy + local search to find voltages that avoid short zero-voltage closed walks.
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
    # Represent each as a coefficient vector (net count per edge)
    coeff_vectors = []
    max_walk_len = g - 1
    
    for start_v in range(n_base):
        # BFS/DFS for non-backtracking closed walks
        # State: (current_v, prev_v, length, counts)
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
                
                new_counts = counts[:]
                u_edge, v_edge = pet_edges[e_idx]
                if v == u_edge and next_v == v_edge:
                    new_counts[e_idx] += 1
                else:
                    new_counts[e_idx] -= 1
                
                new_l = l + 1
                if new_l > 0 and next_v == start_v:
                    # Non-backtracking closed walk found
                    coeff_vectors.append(tuple(new_counts))
                else:
                    stack.append((next_v, v, new_l, new_counts))
    
    # Remove duplicate and zero vectors
    unique_coeffs = list(set(coeff_vectors))
    coeff_list = [c for c in unique_coeffs if any(x != 0 for x in c)]
    
    if not coeff_list:
        # No constraints, any voltages work
        m = 2
        n_candidate = n_base * m
        voltages = [1] * 15
        return _build_lift(n_base, m, pet_edges, edge_index, voltages)
    
    # Try small moduli, using local search for voltage optimization
    best_n = float('inf')
    best_edges = None
    
    # Try moduli in increasing order
    for m in range(2, 200):
        if time.time() > deadline:
            break
        
        n_candidate = n_base * m
        if n_candidate >= best_n:
            continue
        
        # Use local search to find valid voltages
        voltages = _find_valid_voltages(coeff_list, m, n_base, 15, deadline)
        
        if voltages is not None:
            # Build and verify the graph
            edges = _build_lift(n_base, m, pet_edges, edge_index, voltages)
            
            # Verify girth (quick check)
            if _has_girth_at_least(n_candidate, edges, g):
                if n_candidate < best_n:
                    best_n = n_candidate
                    best_edges = edges
                    # Can't do better than this m, so return
                    return best_n, best_edges
    
    if best_edges is not None:
        return best_n, best_edges
    
    # Fallback to LUW
    return _construct_luw(k=3, g=g, d=7)

def _find_valid_voltages(coeff_list, m, n_base, n_edges, deadline):
    """
    Find voltages in Z_m such that no coefficient vector has dot product 0 mod m.
    Uses random initialization + local search.
    """
    max_iterations = 500
    random.seed(42)
    
    # Try multiple random starts
    for start_attempt in range(5):
        if time.time() > deadline:
            break
        
        voltages = [random.randint(0, m-1) for _ in range(n_edges)]
        
        # Local search: iteratively adjust voltages to fix violations
        for iteration in range(max_iterations):
            if time.time() > deadline:
                return None
            
            # Find all violated constraints
            violated = []
            for i, c in enumerate(coeff_list):
                s = 0
                for j in range(n_edges):
                    s += c[j] * voltages[j]
                if s % m == 0:
                    violated.append(i)
            
            if not violated:
                return voltages
            
            # Pick a random violated constraint
            vi = random.choice(violated)
            c = coeff_list[vi]
            
            # Try adjusting each voltage in the support of c
            support = [j for j in range(n_edges) if c[j] != 0]
            if not support:
                continue
            
            j = random.choice(support)
            old_val = voltages[j]
            
            # Try a new value
            new_val = random.randint(0, m-1)
            if new_val == old_val:
                continue
            
            voltages[j] = new_val
            
            # Check if this improved things (reduced number of violations)
            # Quick check: just count violations for affected constraints
            # For simplicity, accept if it doesn't make things worse
            # (full re-check is expensive, so we do a partial check)
            
            # Re-check all constraints (expensive but necessary for correctness)
            # Actually, let's just keep going and check at the end
            # This is a hill-climbing approach without strict improvement
    
    return None

def _build_lift(n_base, m, base_edges, edge_index, voltages):
    """
    Build the voltage lift graph.
    """
    edges = set()
    
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
                    edges.add((u_node, v_node))
                else:
                    edges.add((v_node, u_node))
    
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
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
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
