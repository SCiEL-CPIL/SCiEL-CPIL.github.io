"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _cut_size(labels, adjacency):
    """Compute the number of edges crossing the partition."""
    # For each vertex, count neighbors with opposite label
    # Using matrix multiplication for speed
    # labels is 0/1, opposite is 1-labels
    # Cut size = sum over all edges (i,j) where labels[i] != labels[j]
    # This equals 0.5 * sum_i sum_j adjacency[i,j] * (labels[i] != labels[j])
    # (labels[i] != labels[j]) can be computed as labels[i] + labels[j] - 2*labels[i]*labels[j] for binary
    # Or simply: 1 - 2*labels[i]*(labels[j]-labels[i]) ... let's use direct computation
    # Actually, for binary labels, labels[i] != labels[j] is 1 if they differ.
    # We can compute: for each i, sum of adjacency[i, :] where labels differ.
    # Vectorized: 
    # diff = labels[:, None] != labels[None, :]
    # cut = np.sum(adjacency * diff)
    # But this creates an n x n boolean matrix which is 100x100 = 10000 elements, fine.
    # Alternatively:
    # cut = 0.5 * (labels @ adjacency @ (1 - labels) + (1 - labels) @ adjacency @ labels)
    # Simpler: 
    # For each vertex v, let S_v be sum of adjacency[v, :] where label differs.
    # Total cut = 0.5 * sum_v S_v (since each crossing edge counted twice)
    
    n = len(labels)
    # labels as float for matrix ops
    l = labels.astype(float)
    # For each vertex, the contribution to cut from its perspective:
    # sum of adjacency[v, u] for u where labels[u] != labels[v]
    # = sum_u adjacency[v,u] - sum_u adjacency[v,u] * (labels[u] == labels[v])
    # Let's just do the direct vectorized approach
    diff_matrix = (labels[:, None] != labels[None, :])
    return np.sum(adjacency * diff_matrix)

def partition(adjacency):
    n = len(adjacency)
    deadline = time.time() + 115  # Safety margin before 120s limit
    
    best_labels = None
    best_cut = -1
    
    # Multiple restarts with different initializations
    num_restarts = 50
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Different initialization strategies
        if restart % 3 == 0:
            # Random initialization
            labels = np.random.randint(0, 2, n)
        elif restart % 3 == 1:
            # Alternating
            labels = np.arange(n) % 2
        else:
            # Greedy random: add vertices one by one, assign to side that maximizes cut
            labels = np.zeros(n, dtype=int)
            # Random order
            order = np.random.permutation(n)
            for v in order:
                # Count neighbors already assigned to 0 and 1
                assigned = np.ones(n, dtype=bool)
                assigned[v] = False
                # Actually, we need to track which are assigned
                # Let's use a simpler approach: just random for now
                labels[v] = np.random.randint(0, 2)
        
        # Local improvement with multiple passes until no improvement
        improved = True
        while improved:
            if time.time() > deadline:
                break
            
            improved = False
            # Randomize the order of vertices to avoid bias
            order = np.random.permutation(n)
            for v in order:
                if time.time() > deadline:
                    break
                
                # Count same-side and other-side neighbors
                same = adjacency[v, labels == labels[v]].sum()
                other = adjacency[v, labels != labels[v]].sum()
                
                if same > other:
                    labels[v] = 1 - labels[v]
                    improved = True
        
        # Compute cut size
        cut = _cut_size(labels, adjacency)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    # Final check: if we never found a valid result, use a simple default
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
