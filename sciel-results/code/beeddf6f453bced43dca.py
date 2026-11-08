"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    # For each node, count neighbors with opposite label, sum, divide by 2
    diff = labels[None, :] != labels[:, None]
    return (adjacency & diff).sum() // 2


def _local_search(adjacency, labels, max_time=None):
    """Iterated local search via single-flip improvement."""
    n = len(adjacency)
    labels = labels.copy()
    current_cut = _cut_size(adjacency, labels)
    
    if max_time is None:
        max_time = 145.0
    
    start_time = time.time()
    
    improved = True
    while improved:
        improved = False
        # Shuffle order for diversity
        order = np.random.permutation(n)
        for v in order:
            if max_time is not None and time.time() - start_time > max_time:
                return labels, current_cut
            
            # Count same-side and opposite-side neighbors
            same_count = adjacency[v][labels == labels[v]].sum()
            other_count = adjacency[v][labels != labels[v]].sum()
            
            if same_count > other_count:
                # Flip label
                labels[v] = 1 - labels[v]
                # Update cut size: flipping v changes cut by (other_count - same_count)
                # Actually, each edge to same side becomes crossing, each edge to other side stops crossing
                # So change = other_count - same_count... wait let me think
                # Before flip: edges from v to same side are NOT crossing, edges to other side ARE crossing
                # After flip: edges from v to same side ARE crossing, edges to other side are NOT crossing
                # Change in cut = (edges to same side) - (edges to other side) = same_count - other_count
                current_cut += (same_count - other_count)
                improved = True
    
    return labels, current_cut


def partition(adjacency):
    n = len(adjacency)
    deadline = time.time() + 145.0  # Safety margin before 150s kill
    
    best_labels = None
    best_cut = -1
    
    # Start with alternating initialization
    labels = np.arange(n) % 2
    
    # Multiple restarts with different initializations
    num_restarts = 10
    for restart in range(num_restarts):
        if time.time() > deadline - 10:  # Stop if too close to deadline
            break
        
        if restart == 0:
            # Alternating start
            current_labels = np.arange(n) % 2
        elif restart == 1:
            # Random start
            current_labels = np.random.randint(0, 2, size=n)
        else:
            # Perturbed best solution
            if best_labels is not None:
                current_labels = best_labels.copy()
                # Flip a random subset of vertices
                num_flips = max(1, n // 10)
                flip_indices = np.random.choice(n, size=num_flips, replace=False)
                current_labels[flip_indices] = 1 - current_labels[flip_indices]
            else:
                current_labels = np.random.randint(0, 2, size=n)
        
        # Run local search
        labels, cut = _local_search(adjacency, current_labels, max_time=deadline - 5)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    if best_labels is None:
        # Fallback to simple greedy
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
