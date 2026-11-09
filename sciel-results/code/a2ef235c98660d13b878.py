import numpy as np
import time

def sign_edges(graph, budget):
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))
        
    # Find spanning tree using BFS
    visited = [False] * n
    tree_edge_indices = set()
    queue = [0]
    visited[0] = True
    
    while queue:
        u = queue.pop(0)
        for v, edge_idx in adj[u]:
            if not visited[v]:
                visited[v] = True
                tree_edge_indices.add(edge_idx)
                queue.append(v)
                
    # Identify chords (edges not in tree)
    chords = []
    for i in range(num_edges):
        if i not in tree_edge_indices:
            chords.append(i)
            
    num_chords = len(chords)
    
    # Precompute base matrix with all +1 (tree edges fixed at +1)
    base_A = np.zeros((n, n), dtype=float)
    for i, (u, v) in enumerate(edges):
        if i in tree_edge_indices:
            base_A[u, v] = 1.0
            base_A[v, u] = 1.0
        else:
            base_A[u, v] = 0.0  # placeholder
            base_A[v, u] = 0.0
    
    # Store chord endpoints for quick updating
    chord_endpoints = [(edges[i][0], edges[i][1]) for i in chords]
    
    best_radius = float('inf')
    best_signs = None
    
    # Helper to check if the signed graph is unbalanced (lift is connected)
    def is_unbalanced(signs):
        # Assign potentials to vertices
        # p[v] = 1 or -1
        # p[v] = sign(u,v) * p[u]
        p = [0] * n
        p[0] = 1
        queue = [0]
        visited = [False] * n
        visited[0] = True
        
        while queue:
            u = queue.pop(0)
            for v, edge_idx in adj[u]:
                s = float(signs[edge_idx])
                expected_p_v = s * p[u]
                if not visited[v]:
                    p[v] = expected_p_v
                    visited[v] = True
                    queue.append(v)
                else:
                    # Check consistency
                    if p[v] != expected_p_v:
                        return True # Unbalanced
        return False # Balanced
        
    # Iterate over all masks
    for mask in range(1, 1 << num_chords):
        if time.time() > deadline:
            if best_signs is not None:
                return best_signs
            # Fallback: Find a valid unbalanced sign quickly
            # Try flipping one chord
            for k in range(num_chords):
                signs = np.ones(num_edges, dtype=int)
                signs[chords[k]] = -1
                if is_unbalanced(signs):
                    return signs
            # If all fail (should not happen for connected graph), return any
            return np.ones(num_edges, dtype=int)
                
        signs = np.ones(num_edges, dtype=int)
        
        # Assign signs to chords
        for k, edge_idx in enumerate(chords):
            if mask & (1 << k):
                signs[edge_idx] = -1
            else:
                signs[edge_idx] = 1
                
        # Check validity (connectivity of lift)
        if not is_unbalanced(signs):
            continue
            
        # Build signed adjacency matrix
        A = base_A.copy()
        for k, (u, v) in enumerate(chord_endpoints):
            s = 1.0 if (mask & (1 << k)) == 0 else -1.0
            A[u, v] = s
            A[v, u] = s
            
        # Compute spectral radius
        eigvals = np.linalg.eigvalsh(A)
        radius = np.max(np.abs(eigvals))
        
        if radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
            
    if best_signs is not None:
        return best_signs
    
    # Final fallback
    signs = np.ones(num_edges, dtype=int)
    signs[chords[0]] = -1
    return signs
