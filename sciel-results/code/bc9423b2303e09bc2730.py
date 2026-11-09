import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    best_n = 0
    best_edges = []
    min_edge_count = float('inf')

    # Try small n values. Theoretical min edges is >= 29.
    # n=9 has 126 edges, 512 colorings.
    # n=10 has 252 edges, 1024 colorings.
    # n=11 has 462 edges, 2048 colorings.
    
    for n in range(9, 13):
        if time.time() > deadline:
            break
            
        vertices = list(range(n))
        all_edges = list(combinations(vertices, 5))
        num_colorings = 1 << n
        
        # Precompute edge masks using NumPy for speed
        # For each coloring, determine which edges make it monochromatic
        # A coloring c is monochromatic on edge e if all vertices in e have same color
        # We'll create a matrix: edge_masks[i] is a bitset of colorings that edge i makes monochromatic
        
        # Vectorized approach:
        # For each edge, compute the set of colorings where it's monochromatic
        # Edge e = (v1, v2, v3, v4, v5)
        # Coloring c is monochromatic on e iff:
        #   (c has all 0s on e) OR (c has all 1s on e)
        
        edge_masks = np.zeros(len(all_edges), dtype=np.uint64)
        
        for i, e in enumerate(all_edges):
            edge_vertex_mask = 0
            for v in e:
                edge_vertex_mask |= (1 << v)
            
            # For each coloring, check if it's monochromatic on this edge
            # Vectorized: create array of all colorings, check condition
            all_colorings = np.arange(num_colorings, dtype=np.uint64)
            # Extract bits for vertices in edge
            vals = (all_colorings & edge_vertex_mask).astype(np.uint64)
            # Monochromatic if vals == 0 or vals == edge_vertex_mask
            mono_mask = (vals == 0) | (vals == edge_vertex_mask)
            # Convert boolean mask to bitset
            # We need to pack this into a uint64
            # For n=9, num_colorings=512, so we need 512 bits = 8 uint64s
            # But we're storing in a single uint64... that won't work for n>=6
            # Let me reconsider the approach
            
            # Actually, let's use a different strategy:
            # Represent the set of covered colorings as a Python set or use multiple uint64s
            # For now, let's use a list of sets for clarity, then optimize
            
        # Alternative: use Python sets for the uncovered colorings
        # This is slower but more flexible for larger n
        
        # Let's try a hybrid: use bit arrays with Python integers (arbitrary precision)
        # Python ints can be arbitrarily large, so we can use them as bitsets
        
        edge_masks_py = []
        for i, e in enumerate(all_edges):
            edge_vertex_mask = 0
            for v in e:
                edge_vertex_mask |= (1 << v)
            
            mask = 0
            for c in range(num_colorings):
                val = c & edge_vertex_mask
                if val == 0 or val == edge_vertex_mask:
                    mask |= (1 << c)
            edge_masks_py.append(mask)
        
        uncovered = (1 << num_colorings) - 1
        selected_edges = []
        
        # Greedy Set Cover with early termination
        while uncovered != 0:
            best_edge_idx = -1
            best_gain = 0
            for i, mask in enumerate(edge_masks_py):
                gain_mask = mask & uncovered
                if gain_mask:
                    gain = gain_mask.bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_edge_idx = i
            
            if best_edge_idx == -1:
                break
                
            selected_edges.append(all_edges[best_edge_idx])
            uncovered &= ~edge_masks_py[best_edge_idx]
            
            if time.time() > deadline:
                break
        
        # If we covered all colorings, it's a valid non-2-colorable hypergraph
        if uncovered == 0 and len(selected_edges) < min_edge_count:
            min_edge_count = len(selected_edges)
            best_n = n
            best_edges = selected_edges

    if best_edges:
        return {'n': best_n, 'edges': [list(e) for e in best_edges]}
    else:
        # Fallback: Return a known valid structure if greedy failed (unlikely)
        # Or just return n=9 full graph (126 edges) which is definitely non-2-colorable
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
