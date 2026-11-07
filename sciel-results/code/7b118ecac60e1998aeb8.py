"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def sign_edges(graph, budget):
    from autoresearch_suite.science_tcs_expansion import check_lift
    
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)
    
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    # Identify edge types
    # Prism graph structure:
    # Top cycle: (0, 1), (1, 2), ..., (m-1, 0)
    # Bottom cycle: (m, m+1), ..., (2m-1, m)
    # Matching: (i, i+m)
    
    top_edges = []
    bottom_edges = []
    match_edges = []
    
    for idx, (u, v) in enumerate(edges):
        if u < m and v < m:
            top_edges.append(idx)
        elif u >= m and v >= m:
            bottom_edges.append(idx)
        else:
            match_edges.append(idx)
            
    # Sort indices to maintain consistent order
    top_edges.sort()
    bottom_edges.sort()
    match_edges.sort()
    
    num_top = len(top_edges)
    num_bottom = len(bottom_edges)
    
    # We fix matching signs to +1.
    # We search over signs for top and bottom cycle edges.
    
    best_signs = None
    best_radius = float('inf')
    
    # Pre-allocate sign array
    signs = np.ones(num_edges, dtype=int)
    for idx in match_edges:
        signs[idx] = 1
        
    # The spectrum of the signed prism graph depends on:
    # 1. Product of top cycle signs (holonomy T)
    # 2. Product of bottom cycle signs (holonomy B)
    # 3. The individual edge signs (which affect the eigenvalues)
    
    # Key insight: For a cycle of length m with signs s_0, ..., s_{m-1},
    # the eigenvalues are 2*cos(2*pi*k/m + phase) where phase depends on holonomy.
    # But the actual eigenvalues of the full prism depend on all edge signs.
    
    # Let's use a smarter approach:
    # 1. First, determine which combinations of (prod_top, prod_bot) are valid (connected)
    #    Valid: prod_top == -1 OR prod_bot == -1
    # 2. For each valid (prod_top, prod_bot), find the sign assignment that minimizes radius.
    
    # Actually, let's just do an exhaustive search but with better optimization.
    
    # Build the base adjacency structure once
    # We'll precompute the edge list for faster matrix building
    
    # For the prism graph, the adjacency matrix has a specific structure.
    # Let's exploit this.
    
    # The signed adjacency matrix A has:
    # - Top cycle: A[i, (i+1)%m] = sign_top[i]
    # - Bottom cycle: A[m+i, m+(i+1)%m] = sign_bottom[i]
    # - Matching: A[i, m+i] = 1 (fixed)
    
    # We can build this matrix quickly.
    
    # Let's iterate over all 2^(2m) sign assignments for top and bottom edges.
    # But we can fix the first top edge to +1 (symmetry).
    
    total_bits = num_top + num_bottom
    
    if total_bits == 0:
        return np.ones(num_edges, dtype=int)
        
    # Fix first edge in combined list to +1
    first_edge_idx = top_edges[0] if num_top > 0 else bottom_edges[0]
    remaining_edge_indices = (top_edges[1:] + bottom_edges) if num_top > 0 else (top_edges + bottom_edges[1:])
    num_remaining = len(remaining_edge_indices)
    
    # Pre-fill signs
    for idx in remaining_edge_indices:
        signs[idx] = 1
    signs[first_edge_idx] = 1
    for idx in match_edges:
        signs[idx] = 1
        
    # Precompute which remaining edges are top vs bottom
    remaining_top = [idx for idx in remaining_edge_indices if idx in top_edges]
    remaining_bottom = [idx for idx in remaining_edge_indices if idx in bottom_edges]
    
    # For faster product calculation, track products incrementally
    # But for simplicity, let's compute directly.
    
    # Build A efficiently:
    # Instead of building from scratch, we can update only the changed entries.
    # But for m<=9, building from scratch is fast enough.
    
    # Let's use a different approach: 
    # The eigenvalues of the signed prism can be computed analytically!
    # 
    # For a signed prism graph, the spectrum is determined by:
    # - The holonomies of the two cycles
    # - The "twists" in the cycles
    #
    # Actually, let's just do the exhaustive search with optimized matrix building.
    
    for mask in range(1 << num_remaining):
        if time.time() > deadline - 2:
            break
            
        # Update signs for remaining edges
        for i, idx in enumerate(remaining_edge_indices):
            if (mask >> i) & 1:
                signs[idx] = -1
            else:
                signs[idx] = 1
        
        # Check connectivity
        prod_top = 1
        for idx in top_edges:
            prod_top *= signs[idx]
            
        prod_bot = 1
        for idx in bottom_edges:
            prod_bot *= signs[idx]
            
        if prod_top == 1 and prod_bot == 1:
            continue # Disconnected
            
        # Build A efficiently
        A = np.zeros((n, n), dtype=float)
        
        # Top cycle
        for i in range(m):
            # Find edge index for top cycle edge (i, (i+1)%m)
            # We need to map from cycle position to edge index
            # This is tricky. Let's just use the precomputed lists.
            pass
        
        # Simpler: just fill from edges list
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            A[u, v] = s
            A[v, u] = s
            
        try:
            eigvals = np.linalg.eigvalsh(A)
            radius = max(abs(eigvals[0]), abs(eigvals[-1]))
        except Exception:
            continue
            
        if radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
            
    if best_signs is None:
        return np.ones(num_edges, dtype=int)
        
    return best_signs

# EVOLVE-BLOCK-END
