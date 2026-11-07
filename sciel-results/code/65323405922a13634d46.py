"""Rewritten program for spectral graph lifts task."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import deque

def sign_edges(graph, budget):
    n = graph['n']
    m = n // 2
    edges = graph['edges']
    num_edges = len(edges)
    
    # Build adjacency list for spanning tree
    adj = [[] for _ in range(n)]
    for idx, (u, v) in enumerate(edges):
        adj[u].append((v, idx))
        adj[v].append((u, idx))
    
    # BFS to find spanning tree
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
    
    # Chord edges
    chord_indices = [i for i in range(num_edges) if i not in tree_edges]
    num_chords = len(chord_indices)
    
    # Generate all valid masks (exclude mask 0 which is all +1 chords)
    # Total masks: 2^num_chords
    # Valid masks: 1 to 2^num_chords - 1
    num_valid = (1 << num_chords) - 1
    
    if num_valid == 0:
        # Should not happen for m >= 1
        return np.ones(num_edges, dtype=np.int8)

    # Create binary matrix for chord signs
    # Shape: (num_valid, num_chords)
    # We need to generate binary representations of numbers 1 to 2^k - 1
    masks = np.arange(1, 1 + num_valid, dtype=np.uint64)
    
    # Convert to binary matrix
    chord_bin = np.zeros((num_valid, num_chords), dtype=np.uint8)
    for i in range(num_chords):
        chord_bin[:, i] = (masks >> i) & 1
        
    # Convert to sign values: 0 -> +1, 1 -> -1
    # sign = 1 - 2 * bin
    chord_signs = 1 - 2 * chord_bin
    chord_signs = chord_signs.astype(np.int8)
    
    # Create full sign matrix
    # Shape: (num_valid, num_edges)
    all_signs = np.ones((num_valid, num_edges), dtype=np.int8)
    
    # Assign chord signs to the correct columns
    all_signs[:, chord_indices] = chord_signs
    
    # Build stack of adjacency matrices
    # Shape: (num_valid, n, n)
    A_stack = np.zeros((num_valid, n, n), dtype=np.float64)
    
    # Fill A_stack
    for idx, (u, v) in enumerate(edges):
        s = all_signs[:, idx]
        A_stack[:, u, v] = s
        A_stack[:, v, u] = s
        
    # Compute eigenvalues
    eigs = np.linalg.eigvalsh(A_stack)
    
    # Compute spectral radius for each
    radii = np.max(np.abs(eigs), axis=-1)
    
    # Find the index with minimum radius
    min_idx = np.argmin(radii)
    
    best_signs = all_signs[min_idx]
    
    return best_signs
# EVOLVE-BLOCK-END
