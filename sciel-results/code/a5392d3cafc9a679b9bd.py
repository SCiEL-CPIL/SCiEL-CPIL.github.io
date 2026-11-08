"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=np.float64)
    
    deadline = time.time() + 110  # 10s safety margin before 120s limit
    
    # --- Spectral initialization via Laplacian eigenvector ---
    # L = D - A, find eigenvector corresponding to smallest non-zero eigenvalue
    # For simplicity and speed, use the second-smallest eigenvector (Fiedler vector)
    # But computing full eigendecomposition of 100x100 is fast enough.
    
    # Build Laplacian
    degree = adj.sum(axis=1)
    L = np.diag(degree) - adj
    
    # Get eigenvalues and eigenvectors
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        # The smallest eigenvalue is 0 (for connected graph); use the next one
        # Sort is already done by eigh (ascending)
        fiedler = eigvecs[:, 1]  # second eigenvector
        labels = (fiedler >= 0).astype(np.int8)
    except Exception:
        # Fallback to alternating
        labels = np.arange(n) % 2
    
    # Ensure roughly balanced (if one side is empty, flip to alternate)
    if labels.sum() == 0 or labels.sum() == n:
        labels = np.arange(n) % 2
    
    # --- Vectorized local search: batch flip all vertices that benefit ---
    def _cut_size(labels):
        # Number of edges crossing the cut
        # For each vertex, count neighbors in opposite partition
        # Use matrix multiplication trick
        cross = (labels[:, None] != labels[None, :])
        return int((adj * cross).sum() // 2)
    
    best_labels = labels.copy()
    best_cut = _cut_size(labels)
    
    # Vectorized local search: repeatedly find all vertices that would improve
    # if flipped, flip them all at once, until no improvement
    for iteration in range(50):
        if time.time() > deadline:
            break
        
        # For each vertex v: gain from flipping = 2*same_side_neighbors - degree[v]
        # If gain > 0, flipping improves the cut
        # same_side_neighbors[v] = sum of adj[v, u] where labels[u] == labels[v]
        
        # Vectorized: for each v, count neighbors with same label
        # same_count[v] = sum_u adj[v,u] * (labels[u] == labels[v])
        same_mask = (labels[:, None] == labels[None, :])  # n x n boolean
        same_count = (adj * same_mask).sum(axis=1)  # n vector
        
        # degree is same_count + other_count
        # gain from flipping v = other_count[v] - same_count[v]
        # other_count = degree - same_count
        # gain = degree - 2*same_count
        gain = degree - 2.0 * same_count
        
        # Flip all vertices with positive gain
        to_flip = gain > 0
        if not to_flip.any():
            break
        
        labels[to_flip] = 1 - labels[to_flip]
        
        new_cut = _cut_size(labels)
        if new_cut > best_cut:
            best_cut = new_cut
            best_labels = labels.copy()
    
    # --- Second phase: sequential fine-tuning for remaining single-vertex improvements ---
    if time.time() < deadline:
        for _ in range(10):
            if time.time() > deadline:
                break
            improved = False
            for v in range(n):
                same = adj[v, labels == labels[v]].sum()
                other = adj[v, labels != labels[v]].sum()
                if other > same:
                    labels[v] = 1 - labels[v]
                    improved = True
            if not improved:
                break
            new_cut = _cut_size(labels)
            if new_cut > best_cut:
                best_cut = new_cut
                best_labels = labels.copy()
    
    return best_labels
# EVOLVE-BLOCK-END
