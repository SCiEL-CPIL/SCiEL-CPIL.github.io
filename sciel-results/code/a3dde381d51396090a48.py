import numpy as np
import time

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 50.0
    
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    
    # Identify edge types
    # Vertical edges: (i, i+m)
    # Top cycle: (i, i+1) for i in 0..m-1
    # Bottom cycle: (i+m, i+m+1) for i in 0..m-1
    
    vertical_indices = []
    top_indices = []
    bottom_indices = []
    
    # Map edge endpoints to identify type
    # Note: edges are undirected, but we can check endpoints
    for i, (u, v) in enumerate(edges):
        if abs(u - v) == m:
            vertical_indices.append(i)
        elif u < m and v < m:
            top_indices.append(i)
        elif u >= m and v >= m:
            bottom_indices.append(i)
        else:
            # Should not happen for prism
            pass
            
    # Ensure we have the right counts
    # There are m vertical, m top, m bottom edges
    
    # Fix vertical signs to +1
    signs = np.ones(len(edges), dtype=int)
    
    # We will vary top and bottom cycle signs.
    # Switching symmetry: We can fix one top edge sign to +1.
    # Let's say top_indices[0] is fixed to +1.
    # The remaining 2m - 1 edges are free.
    
    free_top = top_indices[1:]
    free_bottom = bottom_indices[:]
    
    num_free = len(free_top) + len(free_bottom)
    
    best_radius = float('inf')
    best_signs = None
    
    # Iterate over all 2^num_free masks
    for mask in range(1 << num_free):
        if time.time() > deadline:
            break
            
        # Assign signs
        # free_top[k] gets bit k
        # free_bottom[k] gets bit m-1+k
        
        for k, idx in enumerate(free_top):
            signs[idx] = -1 if (mask & (1 << k)) else 1
            
        for k, idx in enumerate(free_bottom):
            # Offset by len(free_top)
            bit_pos = len(free_top) + k
            signs[idx] = -1 if (mask & (1 << bit_pos)) else 1
            
        # Check connectivity:
        # Lift is disconnected iff all square products are +1.
        # Square i product: top[i] * bottom[i] * vertical[i] * vertical[i+1]
        # Verticals are all +1, so product is top[i] * bottom[i].
        # We need at least one top[i] * bottom[i] == -1.
        
        connected = False
        for i in range(m):
            t_sign = signs[top_indices[i]]
            b_sign = signs[bottom_indices[i]]
            if t_sign * b_sign == -1:
                connected = True
                break
        
        if not connected:
            continue
            
        # Build adjacency matrix
        A = np.zeros((n, n), dtype=float)
        for i, (u, v) in enumerate(edges):
            s = signs[i]
            A[u, v] = s
            A[v, u] = s
            
        # Compute spectral radius
        eigvals = np.linalg.eigvalsh(A)
        radius = np.max(np.abs(eigvals))
        
        if radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
            
    if best_signs is None:
        # Fallback: all +1 is invalid (disconnected), so we need a valid one.
        # If loop didn't finish or found nothing, return a known valid one.
        # E.g. flip one bottom edge.
        signs = np.ones(len(edges), dtype=int)
        if bottom_indices:
            signs[bottom_indices[0]] = -1
        best_signs = signs
        
    return best_signs
