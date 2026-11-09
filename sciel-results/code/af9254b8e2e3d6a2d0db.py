import time
import itertools
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    n = 16
    all_edges = []
    edge_to_idx = {}
    for u in range(n):
        for v in range(u + 1, n):
            edge_to_idx[(u, v)] = len(all_edges)
            all_edges.append((u, v))
    
    m = len(all_edges) # 120

    # Fallback: Turan graph T(16, 3)
    # Partitions sizes: 5, 5, 6
    # Edges between partitions.
    # This is definitely Q3-free (triangle-free).
    # Edges: 5*5 + 5*6 + 5*6 = 25 + 30 + 30 = 85.
    def get_turan_edges():
        parts = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14, 15]]
        edges = []
        for i in range(3):
            for j in range(i+1, 3):
                for u in parts[i]:
                    for v in parts[j]:
                        edges.append([u, v])
        return edges

    # Try to build Conflict Graph
    # If it takes too long, return Turan
    conflict_set = set()
    
    # Precompute local Q3 edge sets (28-bit masks)
    # Q3 on 0..7
    q3_local_edges = []
    for i in range(8):
        for j in range(i+1, 8):
            diff = i ^ j
            if diff and (diff & (diff - 1)) == 0:
                q3_local_edges.append((i, j))
                
    # Generate 840 unique local Q3 edge sets (as frozensets of 28-bit indices)
    # We map local edge (i,j) to a bit position 0..27
    local_edge_to_bit = {}
    bit = 0
    for i in range(8):
        for j in range(i+1, 8):
            local_edge_to_bit[(i,j)] = bit
            bit += 1
            
    std_q3_masks = set()
    for p in itertools.permutations(range(8)):
        mask = 0
        for i in range(8):
            for j in range(i+1, 8):
                if (i, j) in local_edge_to_bit:
                    # Check if this edge is in Q3
                    diff = i ^ j
                    if diff and (diff & (diff - 1)) == 0:
                        u, v = p[i], p[j]
                        if u > v: u, v = v, u
                        mask |= (1 << local_edge_to_bit[(u,v)])
        std_q3_masks.add(mask)
        
    # std_q3_masks has 840 elements
    
    # Build Conflict Graph
    # Iterate 8-sets
    # To speed up, we can break early if time is up
    for combo in itertools.combinations(range(n), 8):
        if time.time() > deadline - 30:
            # Precomputation taking too long, fallback to Turan
            return get_turan_edges()
            
        # Map local 0..7 to global
        # We need to map the 28 local edges to global edge indices
        # local_edge_to_global_idx = {}
        # for i in range(8):
        #   for j in range(i+1, 8):
        #     gu, gv = combo[i], combo[j]
        #     if gu > gv: gu, gv = gv, gu
        #     local_edge_to_global_idx[(i,j)] = edge_to_idx[(gu, gv)]
        
        # For each of the 840 Q3 masks:
        for mask in std_q3_masks:
            # Extract the 12 set bits
            # This is slow. 
            # Optimization: Precompute the list of 12 local edges for each mask?
            # There are 840 masks.
            # Let's precompute mask -> list of 12 local edges (i,j)
            pass
            
    # The above loop is too slow to precompute mask->edges inside.
    # Let's precompute mask -> list of 12 local edges (i,j) once.
    mask_to_edges = {}
    for mask in std_q3_masks:
        edges = []
        for i in range(8):
            for j in range(i+1, 8):
                if mask & (1 << local_edge_to_bit[(i,j)]):
                    edges.append((i,j))
        mask_to_edges[mask] = edges
        
    # Now build conflict graph
    conflict_set = set()
    for combo in itertools.combinations(range(n), 8):
        if time.time() > deadline - 30:
            return get_turan_edges()
            
        # Map local to global
        local_to_global = [0]*8
        for i in range(8):
            local_to_global[i] = combo[i]
            
        for mask, local_edges in mask_to_edges.items():
            global_edges = []
            for (u, v) in local_edges:
                gu = local_to_global[u]
                gv = local_to_global[v]
                if gu > gv: gu, gv = gv, gu
                global_edges.append(edge_to_idx[(gu, gv)])
                
            # Add conflicts
            for i in range(12):
                for j in range(i+1, 12):
                    e1 = global_edges[i]
                    e2 = global_edges[j]
                    if e1 > e2: e1, e2 = e2, e1
                    conflict_set.add((e1, e2))
                    
    # If we got here, we have the conflict set
    # Build adjacency list/mask
    conflict_masks = [0] * m
    for (e1, e2) in conflict_set:
        conflict_masks[e1] |= (1 << e2)
        conflict_masks[e2] |= (1 << e1)
        
    # Solve MIS using Greedy + Local Search
    # Greedy: Pick random order, take if no conflict with taken
    best_size = 0
    best_set = 0
    
    # Initial Greedy
    order = list(range(m))
    random.shuffle(order)
    current_set = 0
    current_size = 0
    for v in order:
        if (current_set & conflict_masks[v]) == 0:
            current_set |= (1 << v)
            current_size += 1
            
    best_set = current_set
    best_size = current_size
    
    # Local Search: Try to swap out one edge and in two? Or just add?
    # Simple: Try to add any edge that fits.
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Try to add edges
        for v in range(m):
            if not (current_set & (1 << v)):
                if (current_set & conflict_masks[v]) == 0:
                    current_set |= (1 << v)
                    current_size += 1
                    improved = True
        # Try to swap: remove one, add two?
        # Skip for simplicity, greedy add is often good enough if order is good.
        # Let's do a few random restarts of greedy
        if not improved:
            for _ in range(100):
                if time.time() > deadline - 5: break
                order = list(range(m))
                random.shuffle(order)
                g_set = 0
                g_size = 0
                for v in order:
                    if (g_set & conflict_masks[v]) == 0:
                        g_set |= (1 << v)
                        g_size += 1
                if g_size > best_size:
                    best_size = g_size
                    best_set = g_set
                    
    result_edges = []
    for i in range(m):
        if (best_set >> i) & 1:
            result_edges.append(list(all_edges[i]))
            
    return result_edges

if __name__ == "__main__":
    edges = construct()
    print(len(edges))
