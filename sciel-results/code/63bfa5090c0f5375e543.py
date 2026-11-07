def sign_edges(graph, budget):
    import numpy as np
    import time
    from itertools import product

    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    n = graph['n']
    edges = graph['edges']
    m = n // 2
    E = len(edges)

    # Identify edge indices for bottom cycle, top cycle, and matching
    # Bottom cycle: vertices 0..m-1
    # Top cycle: vertices m..2m-1
    # Matching: (i, i+m)
    
    bot_edge_indices = []
    top_edge_indices = []
    match_edge_indices = []
    
    for i, (u, v) in enumerate(edges):
        if u < m and v < m:
            bot_edge_indices.append(i)
        elif u >= m and v >= m:
            top_edge_indices.append(i)
        else:
            match_edge_indices.append(i)
            
    # Ensure we have the right counts
    # In a standard prism, there are m bottom edges, m top edges, m matching edges.
    # The order of edges in the input list might vary, but the structure is fixed.
    
    # Function to check if a signed graph is unbalanced
    # A signed graph is balanced if every cycle has product +1.
    # It is unbalanced if there exists a cycle with product -1.
    # For a graph with a cycle basis, we just need to check the basis cycles.
    # For a prism, we can check the bottom cycle, top cycle, and one "side" cycle.
    # Actually, simpler: BFS with potentials. If we can assign potentials $p_v \in \{1, -1\}$ such that $s_{uv} = p_u p_v$ for all edges, it's balanced.
    # If no such assignment exists, it's unbalanced.
    
    def is_unbalanced(signs):
        # signs is an array of +1/-1 for each edge
        # Build adjacency with edge indices
        adj = [[] for _ in range(n)]
        for i, (u, v) in enumerate(edges):
            adj[u].append((v, i))
            adj[v].append((u, i))
            
        pot = [0] * n
        pot[0] = 1
        stack = [0]
        while stack:
            u = stack.pop()
            pu = pot[u]
            for v, ei in adj[u]:
                s = signs[ei]
                want = pu * s
                if pot[v] == 0:
                    pot[v] = want
                    stack.append(v)
                elif pot[v] != want:
                    return True # Unbalanced
        return False # Balanced (lift would be disconnected)

    def spectral_radius(signs):
        A = np.zeros((n, n))
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            A[u, v] = s
            A[v, u] = s
        w = np.linalg.eigvalsh(A)
        return float(np.max(np.abs(w)))

    best_signs = None
    best_radius = float('inf')
    
    # Strategy:
    # 1. Enumerate all 2^m sign patterns for the bottom cycle.
    # 2. For each pattern, set top cycle signs to be the negative of bottom cycle signs.
    # 3. Set matching signs to +1.
    # 4. Check if unbalanced.
    # 5. Calculate spectral radius.
    # 6. Keep best.
    
    # We need to map the "logical" edge order (0..m-1 for bottom, 0..m-1 for top)
    # to the actual edge indices in the graph.
    # The input edges are an ordered list. We assume the structure is standard:
    # Bottom edges connect i to (i+1)%m.
    # Top edges connect m+i to m+(i+1)%m.
    # Matching connects i to m+i.
    
    # Let's verify the edge indices correspond to this logical order.
    # It's safer to just use the indices found above.
    
    # Create a mapping from logical edge position to actual index
    # Bottom edges: usually i is (i, (i+1)%m)
    # Top edges: usually i is (m+i, m+(i+1)%m)
    # Matching: usually i is (i, m+i)
    
    # We'll just iterate over all 2^m combinations for the bottom cycle signs.
    # We need to assign these signs to the actual bottom edge indices.
    # The order of edges in the list might not match the logical order 0..m-1.
    # However, the spectral radius depends on the cycle structure.
    # If we permute the signs along the cycle, the spectrum is invariant (cyclic shift).
    # So we can just assign the generated signs to the bottom_edge_indices in order.
    
    for mask in range(2 ** m):
        if time.time() > deadline:
            break
            
        # Generate signs for bottom cycle
        bottom_signs = np.zeros(m, dtype=np.int8)
        for j in range(m):
            if (mask >> j) & 1:
                bottom_signs[j] = 1
            else:
                bottom_signs[j] = -1
        
        # Construct full sign vector
        signs = np.ones(E, dtype=np.int8)
        
        # Assign bottom signs
        for k, ei in enumerate(bot_edge_indices):
            signs[ei] = bottom_signs[k]
            
        # Assign top signs (negative of bottom)
        for k, ei in enumerate(top_edge_indices):
            signs[ei] = -bottom_signs[k]
            
        # Matching signs are +1 (already initialized to 1)
        
        # Check validity (unbalanced)
        if not is_unbalanced(signs):
            continue
            
        # Calculate radius
        r = spectral_radius(signs)
        
        if r < best_radius:
            best_radius = r
            best_signs = signs.copy()
            
    # If the simple strategy didn't find a valid unbalanced graph (unlikely for m>=4),
    # or if we want to try other matching signs, we can expand search.
    # But for m=4,5,7,9, there are always unbalanced configurations.
    # Example: Bottom all +1 (product +1), Top all -1 (product (-1)^m).
    # If m is even, Top product +1. Matching all +1.
    # Cycle: Bottom (+1) * Top (+1) * Matching (+1)^2 ... wait.
    # Let's just trust the enumeration.
    
    # Fallback if no valid sign found (should not happen)
    if best_signs is None:
        best_signs = np.ones(E, dtype=np.int8)
        # Force unbalanced by flipping one edge
        best_signs[0] = -1
        if not is_unbalanced(best_signs):
             # Try flipping another
             best_signs[1] = -1
             if not is_unbalanced(best_signs):
                 best_signs[2] = -1

    return np.asarray(best_signs, dtype=np.int8)
