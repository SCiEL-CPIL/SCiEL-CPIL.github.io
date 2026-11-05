import time
import random
from collections import deque
import numpy as np

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    """
    start_time = time.time()
    deadline = start_time + 145.0
    
    if k == 3:
        return _construct_petersen_lift_optimized(g, deadline)
    else:
        return _construct_luw(k, g, d=7)

def _construct_petersen_lift_optimized(g, deadline):
    """
    Optimized search for voltage lift of Petersen graph.
    Tries all m from Moore bound estimate upwards.
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

    # Precompute coefficient vectors for closed walks of length < g
    coeff_vectors = set()
    max_walk_len = g - 1
    
    # Use BFS/DFS to find non-backtracking closed walks
    # We represent the walk by the net count of each edge.
    # State: (current, prev, length, counts_tuple)
    # To save memory, we only store counts for walks that are "promising" or just all of them?
    # For g=16, the number of walks is manageable.
    
    for start_v in range(n_base):
        stack = [(start_v, -1, 0, [0]*15)]
        while stack:
            v, prev_v, l, counts = stack.pop()
            
            if l >= max_walk_len:
                continue
            
            # If we return to start, record the vector
            if l > 0 and v == start_v:
                coeff_vectors.add(tuple(counts))
            
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
                
                stack.append((next_v, v, l + 1, new_counts))
    
    # Filter out zero vectors (which impose no constraint)
    coeff_list = [np.array(c) for c in coeff_vectors if any(x != 0 for x in c)]
    
    best_n = float('inf')
    best_edges = None
    
    # Moore bound for n(3,16) is 512.
    # Lift size is 10*m. So m >= 51.2.
    # Start from m=52.
    min_m = 52
    
    for m in range(min_m, 300):
        if time.time() > deadline - 10:
            break
            
        n_candidate = 10 * m
        if n_candidate >= best_n:
            continue
        
        # Try to find voltages for this m
        # Use local search
        found = _find_voltages_local_search(coeff_list, m, 500)
        
        if found is not None:
            voltages = found
            edges = _build_lifted_graph(pet_edges, edge_index, voltages, m)
            
            # Verify
            if _has_girth_at_least(n_candidate, edges, g):
                if n_candidate < best_n:
                    best_n = n_candidate
                    best_edges = edges
                    # Since we iterate m in increasing order, the first valid one is the best.
                    # We can return immediately?
                    # Yes, because any larger m gives larger n.
                    return n_candidate, edges
                    
    if best_edges is not None:
        return best_n, best_edges
        
    # Fallback
    return _construct_luw(k=3, g=g, d=11)

def _find_voltages_local_search(coeff_list, m, max_iters):
    """
    Finds voltages in Z_m such that no walk has sum 0.
    """
    if not coeff_list:
        return [0]*15
        
    n_edges = 15
    # Precompute matrix A where A[i] is the coefficient vector of walk i
    A = np.array(coeff_list)
    
    best_violations = float('inf')
    best_vol = None
    
    for restart in range(20):
        if time.time() > time.time() + 1: # Dummy, use global deadline? 
             pass
             
        vol = np.random.randint(1, m, n_edges)
        violations = 0
        for i in range(len(A)):
            s = np.dot(A[i], vol)
            if s % m == 0:
                violations += 1
                
        if violations == 0:
            return vol.tolist()
            
        best_violations = min(best_violations, violations)
        best_vol = vol.copy()
        
        # Local search
        for _ in range(max_iters):
            if violations == 0:
                return vol.tolist()
                
            # Pick a random violating walk
            # We need to identify which walks are violating.
            # To speed up, we can just pick a random walk and check.
            # Or maintain a list of violating indices.
            
            # Simple approach: pick random edge, change voltage.
            e_idx = random.randint(0, n_edges - 1)
            old_val = vol[e_idx]
            new_val = random.randint(1, m - 1)
            if new_val == old_val:
                continue
                
            vol[e_idx] = new_val
            
            # Recompute violations? Expensive.
            # Instead, update violations for walks containing edge e_idx.
            # This is complex. 
            # Given small n_edges (15) and small number of walks (likely < 1000), 
            # full recompute is fast.
            
            new_violations = 0
            for i in range(len(A)):
                s = np.dot(A[i], vol)
                if s % m == 0:
                    new_violations += 1
                    
            if new_violations < violations:
                violations = new_violations
                if violations == 0:
                    return vol.tolist()
            else:
                vol[e_idx] = old_val
                
    return best_vol.tolist() if best_violations < float('inf') else None

def _build_lifted_graph(pet_edges, edge_index, voltages, m):
    n_base = 10
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
                    
    return list(set(edges))

def _construct_luw(k, g, d=None, variant=0):
    # Same as before
    if d is None:
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    q = k
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
