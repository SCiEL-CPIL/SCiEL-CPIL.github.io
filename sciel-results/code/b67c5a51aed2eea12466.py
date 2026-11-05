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
        
    # Precompute the "others" for each vertex in each triple it belongs to
    # For a triple (i,j,k) and vertex i, the others are j,k.
    # This avoids recomputing inside the loop.
    # structure: triple_others[t_idx][vertex] = [other1, other2]
    # But vertex can be i, j, or k.
    # Let's just store for each t_idx, the three vertices and for each vertex, the other two.
    # Actually, in the loop we iterate t_idx in triples_with[u].
    # We need to find the other two vertices.
    # We can precompute a list for each (t_idx, vertex) pair?
    # Or just compute it once and store in a dict keyed by (t_idx, vertex)?
    # There are 1140 * 3 = 3420 such pairs.
    
    other_vertices = {}
    for t_idx, (i, j, k) in enumerate(triples):
        other_vertices[(t_idx, i)] = (j, k)
        other_vertices[(t_idx, j)] = (i, k)
        other_vertices[(t_idx, k)] = (i, j)

    def calc_pen(c):
        if c > 2:
            return (c - 2) ** 2
        return 0

    def init_graph():
        # Start with a random graph with p=0.6
        # This is dense enough to have many edges but we will repair/optimize
        adj = np.zeros((n, n), dtype=np.uint8)
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < 0.6:
                    adj[u, v] = 1
                    adj[v, u] = 1
        return adj

    def compute_penalty(adj):
        counts = np.zeros(num_triples, dtype=np.int32)
        total_pen = 0
        for t_idx, (i, j, k) in enumerate(triples):
            c = 0
            for m in range(n):
                if adj[i, m] and adj[j, m] and adj[k, m]:
                    c += 1
            counts[t_idx] = c
            total_pen += calc_pen(c)
        return counts, total_pen

    def count_edges(adj):
        return int(np.sum(adj) // 2)

    # Try a few initializations to find a good starting point
    best_start_adj = None
    best_start_pen = float('inf')
    best_start_edges = 0
    best_start_counts = None
    
    for _ in range(5):
        adj = init_graph()
        counts, pen = compute_penalty(adj)
        edges = count_edges(adj)
        # We want low penalty and high edges
        # If penalty is 0, we have a feasible graph.
        if pen < best_start_pen or (pen == best_start_pen and edges > best_start_edges):
            best_start_pen = pen
            best_start_edges = edges
            best_start_adj = adj
            best_start_counts = counts
            
    adj = best_start_adj
    counts = best_start_counts
    penalty = best_start_pen
    edge_count = best_start_edges
    
    best_edges = edge_count
    best_adj = adj.copy()
    best_penalty = penalty

    start_time = time.time()
    timeout = 115.0
    
    # SA Parameters
    T = 50.0
    cooling = 0.995 # Faster cooling
    min_T = 0.01
    Lambda = 5.0 # Penalty weight
    
    while time.time() - start_time < timeout and T > min_T:
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
            
        is_add = not adj[u, v]
        
        delta_pen = 0
        
        # Update for triples containing u but not v
        for t_idx in triples_with[u]:
            i, j, k = triples[t_idx]
            if v == i or v == j or v == k:
                continue
            
            x, y = other_vertices[(t_idx, u)]
            
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
            
            x, y = other_vertices[(t_idx, v)]
            
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
        
    # If current is infeasible, try to repair
    if penalty > 0:
        # Greedy repair: remove edges that reduce penalty most
        improved = True
        while improved and penalty > 0:
            improved = False
            best_delta = 0
            best_edge = None
            for u in range(n):
                for v in range(u+1, n):
                    if adj[u, v]:
                        # Simulate removal
                        delta_pen = 0
                        for t_idx in triples_with[u]:
                            i, j, k = triples[t_idx]
                            if v == i or v == j or v == k: continue
                            x, y = other_vertices[(t_idx, u)]
                            if adj[x, v] and adj[y, v]:
                                old_c = counts[t_idx]
                                new_c = old_c - 1
                                delta_pen += calc_pen(new_c) - calc_pen(old_c)
                                counts[t_idx] = new_c
                        for t_idx in triples_with[v]:
                            i, j, k = triples[t_idx]
                            if u == i or u == j or u == k: continue
                            x, y = other_vertices[(t_idx, v)]
                            if adj[x, u] and adj[y, u]:
                                old_c = counts[t_idx]
                                new_c = old_c - 1
                                delta_pen += calc_pen(new_c) - calc_pen(old_c)
                                counts[t_idx] = new_c
                                
                        if delta_pen < best_delta:
                            best_delta = delta_pen
                            best_edge = (u, v)
                            
                        # Undo simulation
                        for t_idx in triples_with[u]:
                            i, j, k = triples[t_idx]
                            if v == i or v == j or v == k: continue
                            x, y = other_vertices[(t_idx, u)]
                            if adj[x, v] and adj[y, v]:
                                counts[t_idx] += 1
                        for t_idx in triples_with[v]:
                            i, j, k = triples[t_idx]
                            if u == i or u == j or u == k: continue
                            x, y = other_vertices[(t_idx, v)]
                            if adj[x, u] and adj[y, u]:
                                counts[t_idx] += 1
                            
            if best_edge:
                u, v = best_edge
                adj[u, v] = 0
                adj[v, u] = 0
                edge_count -= 1
                penalty += best_delta
                improved = True
                if penalty == 0 and edge_count > best_edges:
                    best_edges = edge_count
                    best_adj = adj.copy()

    # Use best feasible found
    if penalty > 0:
        adj = best_adj
        edge_count = best_edges

    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if adj[u, v]:
                edges.append([u, v])
                
    return edges
