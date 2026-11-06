"""
Strategy: Spectral Initialization + Greedy Local Search.
Instead of starting with a simple alternating pattern, we use the top eigenvector 
of the adjacency matrix to initialize the partition. This leverages global graph 
structure (spectral clustering principle) to find a high-quality initial solution, 
which is then polished by local search. This is a different design family from 
pure local search or simulated annealing.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def partition(adjacency):
    """
    Assign each vertex a 0/1 label to maximize the number of cut edges.
    Uses spectral initialization (top eigenvector of adjacency matrix) 
    followed by greedy local search.
    """
    adj = np.asarray(adjacency, dtype=np.float64)
    n = len(adj)
    
    if n <= 1:
        return np.zeros(n, dtype=np.int8)
    
    start_time = time.time()
    deadline = start_time + 100.0  # Leave 20s safety margin
    
    # 1. Spectral Initialization
    # The top eigenvector of the adjacency matrix is a strong heuristic for Max Cut.
    try:
        # Use eigh for symmetric matrices (adjacency is symmetric)
        eigenvalues, eigenvectors = np.linalg.eigh(adj)
        # The largest eigenvalue corresponds to the last column in eigh output
        top_eigenvector = eigenvectors[:, -1]
        
        # Ensure consistent orientation (sign doesn't matter for the cut, but let's be deterministic)
        if top_eigenvector[np.argmax(np.abs(top_eigenvector))] < 0:
            top_eigenvector = -top_eigenvector
            
        # Initialize labels based on sign of eigenvector
        labels = (top_eigenvector > 0).astype(np.int8)
        
    except Exception:
        # Fallback to alternating start if spectral method fails
        labels = np.arange(n) % 2

    # 2. Greedy Local Search
    # We run multiple passes of single-vertex flips.
    # Vectorized gain calculation for speed.
    
    def improve(labels, time_limit):
        # Precompute current cut size
        cut_size = 0
        for i in range(n):
            # Edges crossing: neighbors with different label
            cut_size += np.sum(adj[i] * (labels[i] != labels))
        
        improved = True
        while improved:
            if time.time() > time_limit:
                break
            improved = False
            # Try flipping each vertex
            # Gain of flipping v: 
            # Current contribution of v to cut: edges to opposite side
            # New contribution: edges to same side (which become opposite)
            # Gain = (edges to same side) - (edges to opposite side)
            # = 2 * (edges to same side) - degree(v)
            
            # Vectorized computation of gains for all vertices
            # same_side_count[i] = number of neighbors j where labels[j] == labels[i]
            # We can compute this using matrix multiplication or broadcasting.
            # Let's use broadcasting for clarity, n=100 is small enough.
            
            # labels shape (n, 1), adj shape (n, n)
            # same_mask[i, j] is True if labels[i] == labels[j]
            # This creates an n x n boolean matrix.
            same_mask = (labels[:, None] == labels[None, :])
            
            # Count same-side neighbors for each vertex
            same_counts = np.sum(adj * same_mask, axis=1)
            
            # Degrees
            degrees = np.sum(adj, axis=1)
            
            # Gain for flipping vertex i:
            # If we flip i, its 'same' neighbors become 'opposite' (gain +1 each for cut)
            # Its 'opposite' neighbors become 'same' (loss -1 each for cut)
            # New Cut contribution from i = same_counts[i]
            # Old Cut contribution from i = degrees[i] - same_counts[i]
            # Gain = same_counts[i] - (degrees[i] - same_counts[i]) = 2 * same_counts[i] - degrees[i]
            
            gains = 2 * same_counts - degrees
            
            # Find the vertex with the maximum positive gain
            max_gain = np.max(gains)
            if max_gain > 1e-9:
                flip_idx = np.argmax(gains)
                labels[flip_idx] = 1 - labels[flip_idx]
                improved = True
        
        return labels

    # Run local search with a time budget
    # We allow the local search to run for up to 80 seconds.
    labels = improve(labels, deadline - 20.0)

    # 3. Multi-restart if time remains
    # If we finished early, try a few random restarts to escape local optima.
    current_best_cut = np.sum(adj * (labels[:, None] != labels[None, :])) / 2.0
    best_labels = labels.copy()
    
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        num_restarts = int(remaining_time / 5.0) # ~5 seconds per restart
        for r in range(num_restarts):
            if time.time() > deadline - 2.0:
                break
            
            # Random initialization
            random_labels = np.random.randint(0, 2, n).astype(np.int8)
            
            # Run a shorter local search for this restart
            # Limit time per restart to ensure we finish
            restart_deadline = min(deadline - 2.0, time.time() + 4.0)
            candidate_labels = improve(random_labels, restart_deadline)
            
            # Calculate cut size
            candidate_cut = np.sum(adj * (candidate_labels[:, None] != candidate_labels[None, :])) / 2.0
            
            if candidate_cut > current_best_cut:
                current_best_cut = candidate_cut
                best_labels = candidate_labels

    return best_labels
# EVOLVE-BLOCK-END
