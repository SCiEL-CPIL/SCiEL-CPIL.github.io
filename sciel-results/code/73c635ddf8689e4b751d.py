"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    # labels is a 0/1 array of length n
    # An edge (i,j) crosses if labels[i] != labels[j]
    # For undirected graph, each crossing edge is counted once in the adjacency matrix
    # But we need to be careful: if adjacency[i][j] and adjacency[j][i] are both 1,
    # we count the edge once.
    # Actually, for an undirected graph represented as a boolean matrix,
    # the cut size is sum over i<j where labels differ of adjacency[i][j]
    # But a faster way: for each vertex v, count neighbors in the other set.
    # Since each crossing edge is counted twice (once from each endpoint),
    # we divide by 2.
    n = len(labels)
    # For each vertex, count neighbors in the opposite set
    # Use matrix multiplication: adjacency @ (1 - 2*labels) gives signed degree
    # Actually, let's do it directly
    other = 1 - labels
    # Sum over neighbors in the other set
    cross = np.sum(adjacency * other[None, :])
    # Each crossing edge is counted twice (once from each endpoint)
    return cross // 2


def _improve(adjacency, labels, max_iter=100):
    """Greedy local improvement: flip a vertex if it increases the cut size."""
    n = len(labels)
    labels = labels.copy()
    current_cut = _cut_size(adjacency, labels)
    
    for _ in range(max_iter):
        improved = False
        for v in range(n):
            # If we flip v, how does the cut change?
            # Current contribution of v: edges from v to other set
            other = 1 - labels
            current_v = adjacency[v] @ other
            # If we flip v, v goes to the other set
            # New contribution: edges from v to current set (which is now the other set)
            same = adjacency[v] @ labels
            # Change in cut: (new_v_contribution - old_v_contribution) but careful
            # Actually, flipping v changes:
            # - Edges from v to same set become crossing (gain: same)
            # - Edges from v to other set become non-crossing (loss: other)
            # Net change: same - other = same - (degree - same) = 2*same - degree
            # But wait, let's think more carefully
            # Current cut includes edges from v to other set: that's 'other'
            # After flip, cut includes edges from v to same set: that's 'same'
            # But also, for each neighbor u:
            #   If u is in same set as v (before flip), edge (v,u) was not crossing, now it is
            #   If u is in other set (before flip), edge (v,u) was crossing, now it is not
            # So change = same - other
            delta = same - other
            if delta > 0:
                labels[v] = 1 - labels[v]
                current_cut += delta
                improved = True
        if not improved:
            break
    return labels, current_cut


def partition(adjacency):
    """
    Solve Max-Cut using simulated annealing with greedy local improvement.
    
    Args:
        adjacency: bool matrix of shape (n, n)
    
    Returns:
        0/1 label array of length n
    """
    adjacency = np.asarray(adjacency, dtype=np.int8)
    n = len(adjacency)
    
    # Deadline guard: return best found before 140s (10s safety margin)
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Precompute degrees
    degrees = adjacency.sum(axis=1)
    
    # Multiple restarts with different initializations
    best_labels = None
    best_cut = -1
    
    rng = np.random.default_rng(42)
    
    max_restarts = 50  # Will be limited by time
    
    for restart in range(max_restarts):
        if time.time() > deadline - 10:
            break
        
        # Random initialization
        labels = rng.integers(0, 2, size=n).astype(np.int8)
        
        # Greedy local improvement
        labels, cut = _improve(adjacency, labels, max_iter=50)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
        
        # Simulated annealing from the improved solution
        if time.time() > deadline - 30:
            break
        
        sa_labels = labels.copy()
        sa_cut = cut
        current_temp = 10.0
        cooling = 0.95
        max_sa_steps = 5000
        
        for step in range(max_sa_steps):
            if time.time() > deadline - 10:
                break
            
            # Pick a random vertex to flip
            v = rng.integers(0, n)
            
            # Compute change in cut if we flip v
            other = 1 - sa_labels
            same = adjacency[v] @ sa_labels
            other_count = adjacency[v] @ other
            delta = same - other_count
            
            # Accept if improvement, or with probability exp(-delta/temp) if worsening
            if delta > 0 or rng.random() < np.exp(-delta / current_temp):
                sa_labels[v] = 1 - sa_labels[v]
                sa_cut += delta
            
            # Cool down
            current_temp *= cooling
            
            # Early stop if temperature is very low
            if current_temp < 0.01:
                break
        
        # Final greedy improvement from SA result
        sa_labels, sa_cut = _improve(adjacency, sa_labels, max_iter=30)
        
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()
    
    if best_labels is None:
        # Fallback: alternating pattern
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
