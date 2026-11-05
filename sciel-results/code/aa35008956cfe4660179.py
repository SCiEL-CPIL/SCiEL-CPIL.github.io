import numpy as np
import random
import math
import time

def construct():
    n = 20
    
    # Precompute triples
    triples = []
    triple_index = {}
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                triples.append((i, j, k))
                triple_index[(i, j, k)] = idx
                idx += 1
    
    num_triples = len(triples)
    
    # Map vertex to list of triple indices containing it
    triples_with = [[] for _ in range(n)]
    for t_idx, (i, j, k) in enumerate(triples):
        triples_with[i].append(t_idx)
        triples_with[j].append(t_idx)
        triples_with[k].append(t_idx)
    
    # Initialize graph: K_{2,18}
    # Vertices 0, 1 connected to 2..19
    adj = np.zeros((n, n), dtype=np.uint8)
    for a in range(2):
        for b in range(2, n):
            adj[a, b] = 1
            adj[b, a] = 1
            
    edge_count = 36
    
    # Initialize counts
    counts = np.zeros(num_triples, dtype=np.int32)
    penalty = 0
    
    # Helper to calculate penalty contribution
    def calc_pen(c):
        if c > 2:
            return (c - 2) ** 2
        return 0

    # Initialize counts and penalty
    for t_idx, (i, j, k) in enumerate(triples):
        c = 0
        for m in range(n):
            if adj[i, m] and adj[j, m] and adj[k, m]:
                c += 1
        counts[t_idx] = c
        penalty += calc_pen(c)
        
    # SA Parameters
    start_time = time.time()
    timeout = 110.0 # Leave some time for final checks
    
    T = 100.0
    cooling = 0.9999
    min_T = 0.01
    Lambda = 10.0 # Penalty weight
    
    best_edges = edge_count
    best_adj = adj.copy()
    best_penalty = penalty
    
    while time.time() - start_time < timeout and T > min_T:
        # Pick random edge
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
            
        is_add = not adj[u, v]
        
        # Calculate Delta Penalty
        delta_pen = 0
        
        # Update for triples containing u but not v
        for t_idx in triples_with[u]:
            i, j, k = triples[t_idx]
            if v == i or v == j or v == k:
                continue
            # The triple is (u, j, k) or some permutation
            # The other two vertices are the ones in the triple that are not u and not v
            # Since v is not in the triple, the triple contains u and two others.
            # Let the others be x, y.
            # The term for m=v in C_ijk is x_ux x_vx ... wait.
            # C_ijk = sum_m x_im x_jm x_km.
            # If triple is (u, x, y), C = sum_m x_um x_xm x_ym.
            # We are flipping x_uv.
            # The term for m=v is x_uv x_xv x_yv.
            # If we add edge (u,v), x_uv becomes 1.
            # Delta C is 1 if x_xv and x_yv are 1.
            # We need to identify x and y.
            
            # The triple is (i, j, k). One of them is u.
            # The other two are the ones not u and not v.
            # Since v is not in the triple, the other two are just the ones != u.
            others = []
            if i != u: others.append(i)
            if j != u: others.append(j)
            if k != u: others.append(k)
            # others should have length 2
            x, y = others
            
            if is_add:
                if adj[x, v] and adj[y, v]:
                    old_c = counts[t_idx]
                    new_c = old_c + 1
                    delta_pen += calc_pen(new_c) - calc_pen(old_c)
                    counts[t_idx] = new_c
            else:
                if adj[x, v] and adj[y, v]:
                    old_c = counts[t_idx]
                    new_c = old_c - 1
                    delta_pen += calc_pen(new_c) - calc_pen(old_c)
                    counts[t_idx] = new_c
                    
        # Update for triples containing v but not u
        for t_idx in triples_with[v]:
            i, j, k = triples[t_idx]
            if u == i or u == j or u == k:
                continue
            
            others = []
            if i != v: others.append(i)
            if j != v: others.append(j)
            if k != v: others.append(k)
            x, y = others
            
            if is_add:
                if adj[x, u] and adj[y, u]:
                    old_c = counts[t_idx]
                    new_c = old_c + 1
                    delta_pen += calc_pen(new_c) - calc_pen(old_c)
                    counts[t_idx] = new_c
            else:
                if adj[x, u] and adj[y, u]:
                    old_c = counts[t_idx]
                    new_c = old_c - 1
                    delta_pen += calc_pen(new_c) - calc_pen(old_c)
                    counts[t_idx] = new_c
                    
        delta_edges = 1 if is_add else -1
        delta_E = -delta_edges + Lambda * delta_pen
        
        if delta_E < 0 or random.random() < math.exp(-delta_E / T):
            # Accept
            adj[u, v] = 1 if is_add else 0
            adj[v, u] = 1 if is_add else 0
            edge_count += delta_edges
            penalty += delta_pen
            
            if penalty == 0 and edge_count > best_edges:
                best_edges = edge_count
                best_adj = adj.copy()
                best_penalty = 0
        
        T *= cooling
        
    # Ensure we return a feasible graph
    # If current is infeasible, try to repair or use best feasible
    if penalty > 0:
        # Try to remove edges to fix
        # Simple greedy repair
        for _ in range(1000):
            if penalty == 0:
                break
            # Find an edge that reduces penalty
            improved = False
            for u in range(n):
                for v in range(u+1, n):
                    if not adj[u, v]:
                        continue
                    # Check if removing (u,v) reduces penalty
                    # We can simulate
                    # For simplicity, just try all edges once
                    # This is O(E * affected_triples)
                    pass
                    
        # If still infeasible, return best feasible found
        adj = best_adj
        edge_count = best_edges
        
    # Construct edge list
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if adj[u, v]:
                edges.append([u, v])
                
    return edges
