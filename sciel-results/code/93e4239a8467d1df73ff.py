def sign_edges(graph, budget):
    import numpy as np
    import time
    
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    E = len(edges)
    
    tree_edges = set()
    for i in range(m):
        tree_edges.add(frozenset((i, m+i)))
    for i in range(m-1):
        tree_edges.add(frozenset((i, i+1)))
        
    chord_edges = []
    chord_edges.append(frozenset((m-1, 0)))
    for i in range(m):
        chord_edges.append(frozenset((m+i, m+(i+1)%m)))
        
    num_chords = len(chord_edges)
    
    # Precompute A_base (tree edges are +1)
    A_base = np.zeros((n, n))
    # Precompute edge_to_chord_idx: index in edges -> index in chord_edges or -1
    edge_to_chord_idx = [-1] * E
    chord_pairs = [None] * num_chords
    
    for i, (u, v) in enumerate(edges):
        e = frozenset((u, v))
        if e in tree_edges:
            A_base[u, v] = 1
            A_base[v, u] = 1
        else:
            # Find index in chord_edges
            for j, ce in enumerate(chord_edges):
                if e == ce:
                    edge_to_chord_idx[i] = j
                    chord_pairs[j] = (u, v)
                    break
                    
    best_signs = None
    best_radius = float('inf')
    
    # Precompute signs array template?
    # signs[i] is 1 if tree, else depends on pattern.
    
    for pattern in range(1 << num_chords):
        if time.time() > deadline:
            break
            
        # Balanced if all chord signs are +1 (all bits 1)
        if pattern == (1 << num_chords) - 1:
            continue
            
        A = A_base.copy()
        signs = np.ones(E, dtype=np.int8)
        
        for j in range(num_chords):
            s = 1 if (pattern >> j) & 1 else -1
            u, v = chord_pairs[j]
            A[u, v] = s
            A[v, u] = s
            
            # Update signs array for this edge
            # We need the edge index for this chord
            # We can store chord_edge_indices in chord_pairs tuple?
            # Let's store (u, v, edge_idx)
            pass

    # Refined loop
    best_signs = None
    best_radius = float('inf')
    
    # Recreate chord_pairs with edge indices
    chord_info = [] # (u, v, edge_idx)
    for j, ce in enumerate(chord_edges):
        for i, (u, v) in enumerate(edges):
            if frozenset((u, v)) == ce:
                chord_info.append((u, v, i))
                break
                
    for pattern in range(1 << num_chords):
        if time.time() > deadline:
            break
            
        if pattern == (1 << num_chords) - 1:
            continue
            
        A = A_base.copy()
        signs = np.ones(E, dtype=np.int8)
        
        for j in range(num_chords):
            s = 1 if (pattern >> j) & 1 else -1
            u, v, idx = chord_info[j]
            A[u, v] = s
            A[v, u] = s
            signs[idx] = s
            
        w = np.linalg.eigvalsh(A)
        r = float(np.max(np.abs(w)))
        
        if r < best_radius:
            best_radius = r
            best_signs = signs.copy()
            
    if best_signs is None:
        best_signs = np.array([-1] * E, dtype=np.int8)
        
    return best_signs
