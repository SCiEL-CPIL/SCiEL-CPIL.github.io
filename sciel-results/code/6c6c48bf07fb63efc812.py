import numpy as np
import time

def sign_edges(graph, budget):
    n = graph['n']
    edges = graph['edges']
    m = n // 2
    num_edges = len(edges)
    
    # Build adjacency list with edge indices
    adj = {i: [] for i in range(n)}
    for idx, (u, v) in enumerate(edges):
        adj[u].append((v, idx))
        adj[v].append((u, idx))
        
    # Find a spanning tree using BFS
    visited = [False] * n
    tree_edges = set()
    queue = [0]
    visited[0] = True
    
    while queue:
        u = queue.pop(0)
        for v, idx in adj[u]:
            if not visited[v]:
                visited[v] = True
                tree_edges.add(idx)
                queue.append(v)
                
    # Identify non-tree edges
    non_tree_edges = [i for i in range(num_edges) if i not in tree_edges]
    k = len(non_tree_edges)
    
    best_radius = float('inf')
    best_signs = None
    
    # Pre-compute tree edge signs (all +1)
    # We will construct the full sign vector in each iteration
    
    # Map non-tree edge index to its position in the mask
    # non_tree_edges is a list of original edge indices
    
    start_time = time.time()
    deadline = start_time + 55.0 # Safety margin
    
    # Iterate over all 2^k combinations
    for mask in range(1 << k):
        if time.time() > deadline:
            break
            
        # Skip the balanced case (all non-tree edges +1)
        # Assuming bit 0 -> +1, bit 1 -> -1
        # Balanced case is mask == 0 (all bits 0)
        if mask == 0:
            continue
            
        signs = np.ones(num_edges, dtype=int)
        
        # Set non-tree edge signs based on mask
        for i, edge_idx in enumerate(non_tree_edges):
            if (mask >> i) & 1:
                signs[edge_idx] = -1
            else:
                signs[edge_idx] = 1
                
        # Build signed adjacency matrix
        A = np.zeros((n, n))
        for idx, (u, v) in enumerate(edges):
            s = signs[idx]
            A[u, v] = s
            A[v, u] = s
            
        # Compute spectral radius
        eigvals = np.linalg.eigvalsh(A)
        radius = np.max(np.abs(eigvals))
        
        if radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
            
    # Fallback if no valid lift found (should not happen for m>=2)
    if best_signs is None:
        # Return a random valid lift? 
        # For m=4, k=5. mask=1 is valid.
        # Just return the first valid one found in a small search or default
        # Let's just return all +1 except one edge to make it connected
        signs = np.ones(num_edges, dtype=int)
        signs[0] = -1
        best_signs = signs
        
    return best_signs
