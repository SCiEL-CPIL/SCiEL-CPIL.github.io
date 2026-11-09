import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # We will try to find a small non-2-colorable hypergraph.
    # Strategy: Greedy Set Cover on the space of colorings for small n.
    # If n is small enough, we can enumerate all 2^n colorings.
    
    best_edges = None
    best_n = None
    best_count = float('inf')
    
    # Try n from 11 to 15 (2^n grows, but 2^15 = 32768 is still manageable with bitmasks)
    # The known optimum is 51 edges.
    
    for n in range(11, 16):
        if time.time() > deadline:
            break
            
        # Generate all colorings as bitmasks
        # A coloring is an integer from 0 to 2^n - 1.
        # Bit i is the color of vertex i.
        
        # For each edge (tuple of 5 vertices), compute the set of colorings 
        # where this edge is monochromatic.
        
        # An edge e = (v0, v1, v2, v3, v4) is monochromatic in coloring c if:
        # (c[v0] == c[v1] == c[v2] == c[v3] == c[v4] == 0) OR
        # (c[v0] == c[v1] == c[v2] == c[v3] == c[v4] == 1)
        
        # We can precompute for each edge the two bitmasks:
        # mask0: colorings where all 5 vertices are 0
        # mask1: colorings where all 5 vertices are 1
        # S_e = mask0 | mask1
        
        # To speed up, we can just store the list of edges and their masks.
        
        edges = list(combinations(range(n), 5))
        edge_masks = []
        
        # Optimization: Precompute bit positions
        # For a specific edge (v0..v4), the condition "all 0" means bits v0..v4 are 0.
        # This is not a simple contiguous range of bits, so we can't just use a shift.
        # We must iterate or use clever bit manipulation.
        # Since n <= 15, 2^n <= 32768. Iterating 32768 times per edge (462 edges) is 15M ops.
        # This is acceptable in Python if optimized, but might be slow.
        
        # Better: Iterate through colorings 0..2^n-1 once, and for each coloring, 
        # find which edges are monochromatic? No, that's O(2^n * E).
        
        # Let's stick to iterating edges and building masks.
        # We can optimize the mask generation.
        
        # Actually, for n=11, 2^11=2048. 462 edges. 462*2048 ~ 1M checks.
        # For n=15, 2^15=32768. C(15,5)=3003 edges. 3003*32768 ~ 100M checks. 
        # 100M simple operations in Python might take 10-20 seconds. 
        # We have 110 seconds. So we can try up to n=15.
        
        # Let's implement the mask generation efficiently.
        # For a given edge e, mask0 has 1s at positions c where c has 0s at e's indices.
        # We can generate the list of colorings for mask0 and mask1.
        
        # Alternative: 
        # Just use the greedy set cover directly on the "uncovered" set.
        # Initialize uncovered = (1 << (2**n)) - 1 ? No, that's a bitmask of length 2^n.
        # Python integers can handle this.
        
        # Let's try n=11 first. It's very likely to find a good solution quickly.
        
        total_colorings = 1 << n
        uncovered = (1 << total_colorings) - 1  # Bitmask of all colorings
        
        # Precompute edge masks
        # This is the bottleneck.
        # For each edge, we need a bitmask of length 2^n.
        
        # We can compute the mask for an edge (v0, v1, v2, v3, v4) as follows:
        # The colorings where v0..v4 are all 0 are those c where (c & edge_mask_bits) == 0.
        # The colorings where v0..v4 are all 1 are those c where (c & edge_mask_bits) == edge_mask_bits.
        
        # We can build the mask by iterating c from 0 to 2^n - 1.
        # To make it faster, we can precompute for each vertex v, a bitmask M_v where bit c is 1 if c has 1 at v.
        # Then mask0 for edge e = ~(M_v0 | M_v1 | M_v2 | M_v3 | M_v4)
        # mask1 for edge e = M_v0 & M_v1 & M_v2 & M_v3 & M_v4
        # S_e = mask0 | mask1
        
        # Precomputing M_v for n=15:
        # 15 masks, each of size 2^15 bits (4KB). Very small.
        
        vertex_masks = []
        for v in range(n):
            mask = 0
            # Bit c is 1 if (c >> v) & 1 is 1
            # We can build this by looping c, but 32768 * 15 is small.
            # Or use bit tricks.
            for c in range(total_colorings):
                if (c >> v) & 1:
                    mask |= (1 << c)
            vertex_masks.append(mask)
            
        # Now compute S_e for each edge
        edge_sets = []
        for e in edges:
            m0 = ~(vertex_masks[e[0]] | vertex_masks[e[1]] | vertex_masks[e[2]] | vertex_masks[e[3]] | vertex_masks[e[4]])
            m1 = vertex_masks[e[0]] & vertex_masks[e[1]] & vertex_masks[e[2]] & vertex_masks[e[3]] & vertex_masks[e[4]]
            s_e = m0 | m1
            # We only care about bits within total_colorings
            s_e &= (1 << total_colorings) - 1
            edge_sets.append((e, s_e))
            
        # Greedy Set Cover
        selected_edges = []
        while uncovered:
            if time.time() > deadline:
                break
                
            best_edge = None
            best_count = 0
            best_set = None
            
            # Find edge covering most uncovered colorings
            for e, s_e in edge_sets:
                # Count bits in (s_e & uncovered)
                # Python 3.11+ has int.bit_count()
                count = (s_e & uncovered).bit_count()
                if count > best_count:
                    best_count = count
                    best_edge = e
                    best_set = s_e
                    
            if best_count == 0:
                break # No edge covers any remaining coloring? Should not happen if solution exists
                
            selected_edges.append(best_edge)
            uncovered &= ~best_set
            
        if not uncovered:
            # We have a valid hypergraph
            if len(selected_edges) < best_count:
                best_count = len(selected_edges)
                best_edges = selected_edges
                best_n = n
                
        # Optional: Try to remove edges from the found set to see if it's minimal?
        # Greedy doesn't guarantee minimal, but it's a good start.
        # We can try a simple local search: for each edge in selected_edges, 
        # remove it and check if uncovered becomes non-zero.
        
        if best_edges is None and len(selected_edges) < best_count:
             # Even if not perfect, it's a candidate if it covers all? 
             # Wait, the loop breaks if uncovered is not 0. 
             # So if we exit the while loop with uncovered != 0, it's invalid.
             pass

    # If we found a solution for n=11..15, return it.
    # Otherwise, fall back to the known 126-edge solution or try a different n.
    
    if best_edges is not None:
        return {'n': best_n, 'edges': [list(e) for e in best_edges]}
    else:
        # Fallback: Complete graph on 9 vertices (126 edges)
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
