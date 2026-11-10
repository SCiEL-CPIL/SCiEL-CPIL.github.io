import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n = 31
    target_edges = 91
    
    # Build polarity graph of PG(2,5) - this has exactly 90 edges
    q = 5
    points = []
    for x in range(q):
        for y in range(q):
            points.append((1, x, y))
    for z in range(q):
        points.append((0, 1, z))
    points.append((0, 0, 1))
    
    def dot(p, r):
        return (p[0]*r[0] + p[1]*r[1] + p[2]*r[2]) % q
    
    # Build polarity graph edges
    base_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            if dot(points[i], points[j]) == 0:
                base_edges.add((i, j))
    
    all_pairs = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_pairs.add((i, j))
    
    # Efficient C4 counting and delta computation
    def count_c4(adj):
        """Count 4-cycles in graph given by adjacency matrix."""
        A2 = adj @ adj
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] == 0:
                    c = A2[i, j]
                    if c >= 2:
                        count += c * (c - 1) // 2
        return count // 2
    
    def c4_delta_add_edge(adj, u, v):
        """Compute change in C4 count when adding edge (u,v)."""
        # New C4s formed: for each pair (a,b) where a~u, b~v, a!=b, a!=v, b!=u
        # and a,b not adjacent (to avoid triangles being counted)
        # Actually, a C4 using edge (u,v) as one side: u-v-b-a-u where a~u, b~v, a~b
        # But we need a,b non-adjacent to u,v respectively... 
        # Let me think more carefully.
        # A C4 containing edge (u,v) has form u-v-x-y-u where x~v, y~u, x~y, x!=y
        # The number of such C4s = for each x in N(v)\{u}, for each y in N(u)\{v}, 
        #   if x~y and x!=y, count 1
        # But each C4 is counted once per edge, so adding (u,v) creates:
        #   sum over x in N(v)\{u}, y in N(u)\{v} of [x~y and x!=y]
        # Wait, that's not right either. Let me use the diagonal formula.
        # 
        # C4 count = (1/2) * sum_{i<j, i not~j} C(common(i,j), 2)
        # Adding edge (u,v) changes this in two ways:
        # 1. (u,v) was non-adjacent, so it contributed C(common(u,v),2) to the sum before
        #    Now it's adjacent, so it no longer contributes. Change: -C(common(u,v),2)
        # 2. For other pairs (i,j) where i or j is u or v, their common neighbor count changes
        #    - For pair (i,v) where i!=u: common(i,v) increases by 1 if i~u
        #    - For pair (u,j) where j!=v: common(u,j) increases by 1 if j~v
        #    - For pair (i,j) where i!=u,v, j!=u,v: common(i,j) unchanged
        # 
        # This is getting complex. For n=31, let's just recompute.
        adj_new = adj.copy()
        adj_new[u, v] = 1
        adj_new[v, u] = 1
        return count_c4(adj_new) - count_c4(adj)
    
    # Start with polarity graph (90 edges)
    adj = np.zeros((n, n), dtype=np.int32)
    for u, v in base_edges:
        adj[u, v] = 1
        adj[v, u] = 1
    
    current_c4 = count_c4(adj)
    best_c4 = current_c4
    best_adj = adj.copy()
    
    # Add the best edge (one that minimizes C4 increase)
    non_edges = list(all_pairs - base_edges)
    best_edge = None
    best_delta = float('inf')
    for e in non_edges:
        u, v = e
        adj[u, v] = 1
        adj[v, u] = 1
        new_c4 = count_c4(adj)
        delta = new_c4 - current_c4
        adj[u, v] = 0
        adj[v, u] = 0
        if delta < best_delta:
            best_delta = delta
            best_edge = e
    
    if best_edge:
        u, v = best_edge
        adj[u, v] = 1
        adj[v, u] = 1
        current_c4 += best_delta
        if current_c4 < best_c4:
            best_c4 = current_c4
            best_adj = adj.copy()
    
    # Local search with simulated annealing
    # Use edge swaps to try to reduce C4
    current_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 1:
                current_edges.add((i, j))
    
    all_edges_list = list(all_pairs)
    
    temperature = 10.0
    cooling = 0.9995
    min_temp = 0.01
    no_improve_count = 0
    max_no_improve = 500
    
    for iteration in range(500000):
        if time.time() > deadline - 5:
            break
        if temperature < min_temp:
            break
        if no_improve_count > max_no_improve:
            # Restart with higher temperature
            temperature = 10.0
            no_improve_count = 0
            # Random restart: perturb current solution
            for _ in range(5):
                if current_edges:
                    e_to_remove = random.choice(list(current_edges))
                    non_e = [e for e in all_edges_list if e not in current_edges]
                    if non_e:
                        e_to_add = random.choice(non_e)
                        adj[e_to_remove[0], e_to_remove[1]] = 0
                        adj[e_to_remove[1], e_to_remove[0]] = 0
                        adj[e_to_add[0], e_to_add[1]] = 1
                        adj[e_to_add[1], e_to_add[0]] = 1
                        current_edges.remove(e_to_remove)
                        current_edges.add(e_to_add)
            current_c4 = count_c4(adj)
            continue
        
        # Pick random edge to remove
        if not current_edges:
            break
        remove_edge = random.choice(list(current_edges))
        
        # Pick random edge to add
        non_edges_now = [e for e in all_edges_list if e not in current_edges]
        if not non_edges_now:
            break
        add_edge = random.choice(non_edges_now)
        
        # Apply swap
        old_u, old_v = remove_edge
        new_u, new_v = add_edge
        
        adj[old_u, old_v] = 0
        adj[old_v, old_u] = 0
        adj[new_u, new_v] = 1
        adj[new_v, new_u] = 1
        
        new_c4 = count_c4(adj)
        delta = new_c4 - current_c4
        
        if delta < 0 or (temperature > 0 and random.random() < np.exp(-delta / temperature)):
            # Accept
            current_edges.remove(remove_edge)
            current_edges.add(add_edge)
            current_c4 = new_c4
            no_improve_count = 0
            if current_c4 < best_c4:
                best_c4 = current_c4
                best_adj = adj.copy()
        else:
            # Revert
            adj[old_u, old_v] = 1
            adj[old_v, old_u] = 1
            adj[new_u, new_v] = 0
            adj[new_v, new_u] = 0
            no_improve_count += 1
        
        temperature *= cooling
    
    # Extract edges from best_adj
    edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i, j] == 1:
                edges.add((i, j))
    
    # Ensure exactly 91 edges
    if len(edges) != target_edges:
        if len(edges) < target_edges:
            for edge in all_pairs:
                if len(edges) >= target_edges:
                    break
                if edge not in edges:
                    edges.add(edge)
        else:
            edge_list = list(edges)
            random.shuffle(edge_list)
            edges = set(edge_list[:target_edges])
    
    result = [[u, v] for u, v in edges]
    return result
