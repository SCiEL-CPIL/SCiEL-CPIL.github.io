import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # We will try to minimize edges starting from K_9^5
    # n=9, edges = 126
    # n=10, edges = 252 (might take longer, but could yield smaller core?)
    # Let's stick to n=9 for reliability and speed, as 2^9 is very small.
    
    n = 9
    vertices = list(range(n))
    all_edges = [list(e) for e in combinations(vertices, 5)]
    num_edges = len(all_edges)
    
    total_colorings = 1 << n
    all_bits = (1 << total_colorings) - 1
    
    # Precompute masks for each vertex being 0 or 1
    # mask0[v] has bit i set if vertex v is 0 in coloring i
    # mask1[v] has bit i set if vertex v is 1 in coloring i
    mask0 = [0] * n
    mask1 = [0] * n
    
    for c in range(total_colorings):
        for v in range(n):
            if (c >> v) & 1:
                mask1[v] |= (1 << c)
            else:
                mask0[v] |= (1 << c)
                
    # Precompute mask for each edge: set of colorings where edge is monochromatic
    edge_masks = []
    for e in all_edges:
        m0 = all_bits
        m1 = all_bits
        for v in e:
            m0 &= mask0[v]
            m1 &= mask1[v]
        edge_masks.append(m0 | m1)
        
    current_edges_indices = list(range(num_edges))
    
    # Function to check if a set of edge indices forms a non-2-colorable hypergraph
    # Returns True if non-2-colorable (i.e., union of masks covers all_bits)
    def is_non_colorable(indices):
        union_mask = 0
        for idx in indices:
            union_mask |= edge_masks[idx]
        return union_mask == all_bits

    # Greedy edge removal
    # We iterate and try to remove each edge if the rest is still non-colorable
    # We do multiple passes until no more edges can be removed
    
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        # Iterate backwards to allow safe removal
        for i in range(len(current_edges_indices) - 1, -1, -1):
            if time.time() > deadline:
                break
            idx = current_edges_indices[i]
            # Check if removing idx leaves a non-colorable graph
            # To avoid O(N) check every time, we could maintain union, but N is small.
            # However, recomputing union for N-1 edges is O(N).
            # Total complexity per pass O(N^2).
            
            # Optimization: Only check if it's worth it.
            # If current set is non-colorable, removing one edge might make it colorable.
            # We check if union of all OTHER edges == all_bits.
            
            others = current_edges_indices[:i] + current_edges_indices[i+1:]
            if is_non_colorable(others):
                # Safe to remove
                current_edges_indices.pop(i)
                improved = True
                # Restart the loop? Or continue?
                # Continuing is fine, but the list changed.
                # Since we are iterating backwards, popping i is safe.
                # But the "others" calculation was O(N).
                # If we remove many, we might want to restart to be sure.
                # But greedy removal usually converges fast.
                
    final_edges = [all_edges[idx] for idx in current_edges_indices]
    
    # If we have time, try to improve by adding/removing or trying n=10?
    # Given the constraint "Minimize the number of edges", and the known bound is 51,
    # and K_9^5 reduction might yield ~60, we might want to try K_10^5 if fast enough.
    # But K_10^5 has 252 edges. 252^2 * 1024 bits ops. Still fast.
    # Let's try K_10^5 if we finish K_9^5 quickly.
    
    best_edges = final_edges
    best_n = n
    
    if time.time() < deadline - 20.0:
        # Try n=10
        n2 = 10
        vertices2 = list(range(n2))
        all_edges2 = [list(e) for e in combinations(vertices2, 5)]
        num_edges2 = len(all_edges2)
        
        total_colorings2 = 1 << n2
        all_bits2 = (1 << total_colorings2) - 1
        
        mask0_2 = [0] * n2
        mask1_2 = [0] * n2
        
        for c in range(total_colorings2):
            for v in range(n2):
                if (c >> v) & 1:
                    mask1_2[v] |= (1 << c)
                else:
                    mask0_2[v] |= (1 << c)
                    
        edge_masks2 = []
        for e in all_edges2:
            m0 = all_bits2
            m1 = all_bits2
            for v in e:
                m0 &= mask0_2[v]
                m1 &= mask1_2[v]
            edge_masks2.append(m0 | m1)
            
        current_edges_indices2 = list(range(num_edges2))
        
        def is_non_colorable2(indices):
            union_mask = 0
            for idx in indices:
                union_mask |= edge_masks2[idx]
            return union_mask == all_bits2

        improved2 = True
        while improved2:
            if time.time() > deadline - 5.0:
                break
            improved2 = False
            for i in range(len(current_edges_indices2) - 1, -1, -1):
                if time.time() > deadline - 5.0:
                    break
                idx = current_edges_indices2[i]
                others = current_edges_indices2[:i] + current_edges_indices2[i+1:]
                if is_non_colorable2(others):
                    current_edges_indices2.pop(i)
                    improved2 = True
                    
        final_edges2 = [all_edges2[idx] for idx in current_edges_indices2]
        
        if len(final_edges2) < len(best_edges):
            best_edges = final_edges2
            best_n = n2

    return {'n': best_n, 'edges': best_edges}
