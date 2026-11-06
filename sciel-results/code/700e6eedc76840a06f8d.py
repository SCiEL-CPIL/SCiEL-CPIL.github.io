"""Combined program for Maximum Cut: efficient local search with multiple restarts."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def partition(adjacency):
    """
    Assign each vertex a 0/1 label to maximize the number of cut edges.
    Uses efficient gain computation, multiple restarts, and time guarding.
    """
    adj = np.asarray(adjacency, dtype=np.int8)
    n = len(adj)
    
    if n <= 1:
        return np.zeros(n, dtype=np.int8)
    
    # Precompute degrees for efficient gain calculation
    degrees = adj.sum(axis=1)
    
    deadline = time.time() + 115.0  # Safety margin before 150s kill
    
    best_labels = None
    best_cut = -1
    
    def compute_cut(labels):
        """Number of edges crossing the partition."""
        # For each vertex, count neighbors in opposite group
        # Sum over all vertices, divide by 2 (each edge counted twice)
        cut = 0
        for v in range(n):
            if labels[v] == 0:
                cut += adj[v, 1].sum()
            else:
                cut += adj[v, 0].sum()
        return cut // 2
    
    def local_search(labels, max_passes=10):
        """
        Greedy local search: flip vertex if it increases cut size.
        Uses efficient gain computation: gain = 2*other - degree
        where 'other' is number of neighbors in opposite group.
        """
        labels = labels.copy()
        improved = True
        passes = 0
        
        while improved and passes < max_passes:
            improved = False
            passes += 1
            
            for v in range(n):
                if time.time() > deadline - 2.0:
                    return labels, False
                
                # Count neighbors in opposite group
                if labels[v] == 0:
                    other = adj[v, 1].sum()
                else:
                    other = adj[v, 0].sum()
                
                # Gain from flipping: new_cut - old_cut
                # old contribution of v: other (edges from v to opposite group)
                # new contribution of v: degree[v] - other (edges from v to same group become cut)
                # Net gain = (degree[v] - other) - other = degree[v] - 2*other
                gain = degrees[v] - 2 * other
                
                if gain > 0:
                    labels[v] = 1 - labels[v]
                    improved = True
        
        return labels, improved
    
    rng = np.random.default_rng(seed=42)
    
    # Strategy 1: Alternating start
    labels_alt = np.arange(n) % 2
    labels_improved, _ = local_search(labels_alt)
    cut = compute_cut(labels_improved)
    if cut > best_cut:
        best_cut = cut
        best_labels = labels_improved.copy()
    
    # Strategy 2: All-zeros start
    labels_zero = np.zeros(n, dtype=np.int8)
    labels_improved, _ = local_search(labels_zero)
    cut = compute_cut(labels_improved)
    if cut > best_cut:
        best_cut = cut
        best_labels = labels_improved.copy()
    
    # Strategy 3: Random restarts
    for restart in range(200):
        if time.time() > deadline - 5.0:
            break
        
        labels_rand = rng.integers(0, 2, size=n).astype(np.int8)
        labels_improved, _ = local_search(labels_rand)
        cut = compute_cut(labels_improved)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels_improved.copy()
        
        # Early termination if we're doing well and time is running out
        if time.time() > deadline - 3.0 and restart > 20:
            break
    
    if best_labels is None:
        best_labels = np.zeros(n, dtype=np.int8)
    
    return best_labels.astype(np.int8)
# EVOLVE-BLOCK-END
