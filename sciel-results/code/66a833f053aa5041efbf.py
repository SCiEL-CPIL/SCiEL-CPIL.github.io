"""Rewritten program for Maximum Cut task."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def partition(adjacency):
    """
    Solve Maximum Cut using multi-start local search with random restarts and perturbation.
    """
    A = np.array(adjacency, dtype=bool)
    n = len(A)
    if n == 0:
        return np.array([])
        
    degrees = A.sum(axis=1)
    
    def calc_cut(labels):
        """Calculate the size of the cut."""
        # labels: 0/1 array
        # Cut edges are those where labels differ.
        # diff_mask[i, j] is True if labels[i] != labels[j]
        # Since A is symmetric, sum(A * diff_mask) counts each edge twice.
        diff_mask = (labels[:, None] != labels[None, :])
        return int((A * diff_mask).sum() // 2)

    def local_search(labels):
        """
        Perform local search by flipping all vertices that improve the cut individually.
        Repeats until no improvement is found.
        """
        labels = labels.copy()
        # Limit iterations to prevent infinite loops in case of oscillation
        for _ in range(200): 
            # Calculate for each vertex how many neighbors have the same label
            # same_mask[i, j] is True if labels[i] == labels[j]
            same_mask = (labels[:, None] == labels[None, :])
            # Count same-label neighbors for each vertex
            same_counts = (A * same_mask).sum(axis=1)
            # Count diff-label neighbors
            diff_counts = degrees - same_counts
            
            # Identify vertices where flipping would increase the cut
            # Flipping v changes its contribution from diff_counts[v] to same_counts[v]
            # Improvement if same_counts[v] > diff_counts[v]
            to_flip = same_counts > diff_counts
            
            if not to_flip.any():
                break
                
            # Flip all such vertices simultaneously
            labels[to_flip] = 1 - labels[to_flip]
            
        return labels

    start_time = time.time()
    deadline = 115.0  # Safety margin from 120s limit

    best_labels = None
    best_cut = -1

    # 1. Initial heuristic start (alternating)
    labels = np.arange(n) % 2
    labels = local_search(labels)
    cut = calc_cut(labels)
    best_labels = labels
    best_cut = cut

    # 2. Random Restarts and Perturbations
    # We loop until we run out of time
    while time.time() - start_time < deadline:
        # Strategy A: Pure Random Restart
        if np.random.rand() < 0.5:
            # Random initialization
            labels = np.random.randint(0, 2, n)
            labels = local_search(labels)
            cut = calc_cut(labels)
        else:
            # Strategy B: Perturb Best Solution
            # Flip a small number of random vertices in the best solution found so far
            if best_labels is not None:
                labels = best_labels.copy()
                # Flip 5-10% of vertices
                num_flips = max(1, int(n * 0.05))
                flip_indices = np.random.choice(n, num_flips, replace=False)
                labels[flip_indices] = 1 - labels[flip_indices]
                labels = local_search(labels)
                cut = calc_cut(labels)
            else:
                continue

        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()

    return best_labels
# EVOLVE-BLOCK-END
