"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def sign_edges(graph, budget):
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)
    
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    # Identify edge types
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
            
    top_edges.sort()
    bottom_edges.sort()
    match_edges.sort()
    
    num_top = len(top_edges)
    num_bottom = len(bottom_edges)
    
    # Key insight: For prism graphs, the spectrum depends only on
    # the cycle products (holonomies) due to gauge invariance.
    # We can fix gauge: all matching = +1, first top = +1, first bottom = +1
    # Then search only over the remaining 2m-2 edges.
    # 
    # But even better: we can compute the spectral radius analytically
    # from the two cycle products.
    #
    # For a prism graph with top product P_t and bottom product P_b,
    # the eigenvalues of the signed adjacency matrix are:
    #   2*cos(k*pi/m) + P_t*P_b*cos(k*pi/m)  ... wait, let me derive this.
    #
    # Actually, let me just do the exhaustive search but with:
    # 1. Gauge fixing (fix 3 edges to +1)
    # 2. Fast eigenvalue computation using the structure
    #
    # The signed adjacency matrix has a block structure:
    # A = [[B_top, M], [M^T, B_bot]]
    # where B_top is the signed cycle matrix for top, M is the matching.
    #
    # With matching all +1, M = I_m.
    # B_top and B_bot are m x m matrices with 1s on the super/sub-diagonal
    # and the wrap-around entries being the product of the signs.
    #
    # Actually, for a cycle with edge signs s_0, s_1, ..., s_{m-1},
    # the adjacency matrix has:
    #   A[i, i+1] = s_i for i = 0..m-2
    #   A[m-1, 0] = s_{m-1}
    #   A[i+1, i] = s_i
    #   A[0, m-1] = s_{m-1}
    #
    # This is a circulant-like matrix. The eigenvalues are:
    #   lambda_k = s_0 * omega^k + s_1 * omega^{k+1} + ... + s_{m-1} * omega^{k+m-1}
    # where omega = exp(2*pi*i/m)... no, that's not quite right for signed cycles.
    #
    # Let me just do the direct approach: fix gauge, iterate over remaining bits,
    # compute eigvalsh of the 2m x 2m matrix. With gauge fixing, we have
    # 2^(2m-3) iterations (fixing first top, first bottom, and all matchings).
    # For m=9: 2^15 = 32768 iterations. Very fast.
    
    best_signs = None
    best_radius = float('inf')
    
    signs = np.ones(num_edges, dtype=int)
    for idx in match_edges:
        signs[idx] = 1
    
    # Fix gauge: first top edge and first bottom edge to +1
    if num_top > 0:
        signs[top_edges[0]] = 1
    if num_bottom > 0:
        signs[bottom_edges[0]] = 1
    
    # Remaining edges to search over
    remaining = []
    for idx in top_edges[1:]:
        remaining.append(idx)
    for idx in bottom_edges[1:]:
        remaining.append(idx)
    
    num_remaining = len(remaining)
    
    if num_remaining == 0:
        # Only m=1 case, but m>=4 here
        return signs
    
    # Pre-compute edge endpoints for fast matrix building
    edge_u = np.array([u for u, v in edges], dtype=int)
    edge_v = np.array([v for u, v in edges], dtype=int)
    
    # We'll iterate over all 2^num_remaining masks
    # For each mask, set the signs, compute eigvalsh, track best
    
    # Optimization: use numpy to build the matrix efficiently
    # A[u,v] = signs[idx], A[v,u] = signs[idx]
    
    for mask in range(1 << num_remaining):
        if time.time() > deadline - 2:
            break
        
        # Set signs for remaining edges
        for i, idx in enumerate(remaining):
            signs[idx] = -1 if (mask >> i) & 1 else 1
        
        # Quick connectivity check: 
        # The lift is connected iff NOT (prod_top == 1 AND prod_bot == 1)
        # i.e., at least one cycle has an odd number of -1 signs
        prod_top = 1
        for idx in top_edges:
            prod_top *= signs[idx]
        prod_bot = 1
        for idx in bottom_edges:
            prod_bot *= signs[idx]
        
        if prod_top == 1 and prod_bot == 1:
            continue
        
        # Build signed adjacency matrix
        A = np.zeros((n, n), dtype=float)
        # Use numpy indexing for speed
        A[edge_u, edge_v] = signs.astype(float)
        A[edge_v, edge_u] = signs.astype(float)
        
        # Compute spectral radius
        eigvals = np.linalg.eigvalsh(A)
        radius = max(abs(eigvals[0]), abs(eigvals[-1]))
        
        if radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
    
    if best_signs is None:
        # Fallback: try a known valid pattern
        # All top +1, all bottom -1 (or vice versa)
        for idx in bottom_edges:
            signs[idx] = -1
        return signs
    
    return best_signs

# EVOLVE-BLOCK-END
