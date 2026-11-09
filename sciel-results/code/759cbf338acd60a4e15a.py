# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    d = 7
    n = 2 ** d  # 128 vertices
    
    # Enumerate all edges of Q7
    edges = []
    edge_to_idx = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(edges)
                edges.append((x, y))
                edge_to_idx[(x, y)] = idx
                edge_to_idx[(y, x)] = idx
    
    m = len(edges)  # 448
    
    # Build conflict graph adjacency (bitsets)
    # conflict_mask[k] = bitmask of edges that conflict with edge k
    conflict_mask = [0] * m
    
    for idx, (x, y) in enumerate(edges):
        diff = x ^ y
        i = diff.bit_length() - 1
        for j in range(d):
            if j == i:
                continue
            c = x ^ (1 << j)
            e = y ^ (1 << j)
            c_idx = edge_to_idx[(min(c, e), max(c, e))]
            conflict_mask[idx] |= (1 << c_idx)
    
    all_mask = (1 << m) - 1
    
    # comp_adj[k]: edges that can be in a clique with k in the complement graph
    # i.e., edges that do NOT conflict with k
    comp_adj = [all_mask ^ conflict_mask[k] ^ (1 << k) for k in range(m)]
    
    # Pre-compute degrees for ordering
    degrees = [comp_adj[k].bit_count() for k in range(m)]
    
    # Greedy initialization: Find a large clique in comp_adj
    # This gives a good lower bound for best_size
    def greedy_clique_init():
        P = all_mask
        clique = []
        while P:
            # Pick vertex with highest degree in P
            best_v = -1
            best_deg = -1
            tmp = P
            while tmp:
                lowbit = tmp & (-tmp)
                v = lowbit.bit_length() - 1
                # Count neighbors in P
                deg_in_P = (comp_adj[v] & P).bit_count()
                if deg_in_P > best_deg:
                    best_deg = deg_in_P
                    best_v = v
                tmp ^= lowbit
            
            if best_v == -1:
                break
            
            clique.append(best_v)
            # Remove best_v and its non-neighbors from P? 
            # In Max Clique, we keep neighbors.
            # New P = P & comp_adj[best_v]
            P &= comp_adj[best_v]
        
        return clique
    
    init_clique = greedy_clique_init()
    best_size = len(init_clique)
    best_clique = init_clique[:]
    
    # Tomita's Maximum Clique with Bit-Parallelism
    # We use a stack-based recursion or recursive function with pruning.
    
    start_time = time.time()
    TIME_LIMIT = 110.0 # Leave 10s for safety
    
    # Greedy coloring for upper bound
    # Returns list of color classes (bitmasks)
    def greedy_color(P):
        color_classes = []
        remaining = P
        while remaining:
            # Pick a vertex from remaining
            # Heuristic: pick vertex with lowest degree in remaining to color fast?
            # Or just lowest index.
            # For upper bound, any valid coloring works. 
            # Standard Tomita uses a specific ordering.
            # Let's use a simple sequential coloring.
            
            cc = 0
            avail = remaining
            while avail:
                lowbit = avail & (-avail)
                v = lowbit.bit_length() - 1
                cc |= lowbit
                # Remove v and its neighbors from avail
                # Neighbors in the graph being colored (comp_adj)
                avail &= ~(comp_adj[v] | lowbit)
            
            color_classes.append(cc)
            remaining &= ~cc
        
        return color_classes
    
    def max_clique_rec(P, clique, depth):
        nonlocal best_clique, best_size
        
        # Time check
        if time.time() - start_time > TIME_LIMIT:
            return
        
        if depth + P.bit_count() <= best_size:
            return
        
        if P == 0:
            if depth > best_size:
                best_size = depth
                best_clique = clique[:]
            return
        
        # Greedy coloring
        color_classes = greedy_color(P)
        max_color = len(color_classes)
        
        if depth + max_color <= best_size:
            return
        
        # Branch on vertices in reverse color order
        for c in range(max_color - 1, -1, -1):
            if depth + c + 1 <= best_size:
                break
            
            cc = color_classes[c]
            # Iterate through vertices in this color class
            # To improve pruning, we should process them in an order that
            # allows for early pruning.
            
            tmp = cc
            while tmp:
                lowbit = tmp & (-tmp)
                v = lowbit.bit_length() - 1
                
                if depth + 1 + c <= best_size:
                    break
                
                # Include v
                new_P = P & comp_adj[v]
                max_clique_rec(new_P, clique + [v], depth + 1)
                
                # Remove v from P
                P ^= lowbit
                tmp ^= lowbit
                
                # Prune: if P becomes 0, break
                if P == 0:
                    break

    # Start search
    # We can order the initial vertices by degree to improve performance
    # But the recursion handles ordering via coloring.
    max_clique_rec(all_mask, [], 0)
    
    result = []
    for idx in best_clique:
        x, y = edges[idx]
        result.append([x, y])
    
    return result

# EVOLVE-BLOCK-END
