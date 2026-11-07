"""Rewritten program for spectral graph lifts task."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque

def sign_edges(graph, budget):
    n = graph['n']
    m = n // 2
    edges = graph['edges']
    num_edges = len(edges)
    
    # Identify edge types
    # Build adjacency for BFS
    adj = [[] for _ in range(n)]
    for idx, (u, v) in enumerate(edges):
        adj[u].append((v, idx))
        adj[v].append((u, idx))
    
    # Find spanning tree
    visited = [False] * n
    tree_edges = set()
    queue = deque([0])
    visited[0] = True
    while queue:
        u = queue.popleft()
        for v, eidx in adj[u]:
            if not visited[v]:
                visited[v] = True
                tree_edges.add(eidx)
                queue.append(v)
    
    chord_edges = [i for i in range(num_edges) if i not in tree_edges]
    num_chords = len(chord_edges)
    
    # Precompute A_tree (signed adjacency with +1 on tree edges, 0 elsewhere)
    A_tree = np.zeros((n, n))
    for idx, (u, v) in enumerate(edges):
        if idx in tree_edges:
            A_tree[u, v] = 1.0
            A_tree[v, u] = 1.0
            
    # Precompute chord endpoints for fast updates
    chord_u = [edges[i][0] for i in chord_edges]
    chord_v = [edges[i][1] for i in chord_edges]
    
    best_radius = float('inf')
    best_signs = None
    
    # Enumerate all 2^num_chords assignments
    # Note: The assignment where all chords are +1 results in a disconnected lift (trivial lift).
    # We must skip this case or check validity.
    # Validity: Lift is connected iff at least one chord sign is -1.
    
    for mask in range(1 << num_chords):
        # Check validity: if mask == 0, all chords are +1, lift is disconnected.
        if mask == 0:
            continue
            
        # Construct signs array
        signs = np.ones(num_edges, dtype=np.int8)
        
        # Construct A for spectral radius
        # Start with A_tree, then apply chord signs
        A = A_tree.copy()
        
        for i in range(num_chords):
            if (mask >> i) & 1:
                s = -1
                signs[chord_edges[i]] = -1
                A[chord_u[i], chord_v[i]] = -1.0
                A[chord_v[i], chord_u[i]] = -1.0
            else:
                # s = 1
                # A already has 0 here, need to set to 1?
                # Wait, A_tree has 0 on chord edges.
                # So if s=1, we must add 1.
                A[chord_u[i], chord_v[i]] = 1.0
                A[chord_v[i], chord_u[i]] = 1.0
        
        # Compute spectral radius
        eigvals = np.linalg.eigvalsh(A)
        radius = np.max(np.abs(eigvals))
        
        if radius < best_radius:
            best_radius = radius
            best_signs = signs.copy()
            
    if best_signs is None:
        # Fallback: return all +1s (though this is invalid, better than crash)
        # Or find a valid one manually?
        # For m=4, chords=5. Mask 1 is valid.
        # This shouldn't happen.
        return np.ones(num_edges, dtype=np.int8)
        
    return best_signs
# EVOLVE-BLOCK-END
