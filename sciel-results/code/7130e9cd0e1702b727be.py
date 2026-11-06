"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(labels, adjacency):
    """Compute the number of edges crossing the partition."""
    n = len(labels)
    # An edge (i, j) crosses if labels[i] != labels[j].
    # Since adjacency is symmetric and unweighted, we count each crossing edge once.
    # sum over i < j where labels[i] != labels[j] and adjacency[i,j] is True.
    # Vectorized:
    diff = labels[:, None] != labels[None, :]
    # Only count upper triangle to avoid double counting
    triu_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    crossings = diff & triu_mask
    return int((crossings & adjacency).sum())


def _local_search(labels, adjacency, time_limit, rng):
    """
    Perform local search (flip individual vertices) to improve cut size.
    Uses a time limit to stop before exceeding the budget.
    """
    n = len(labels)
    labels = labels.copy()
    adj = adjacency.astype(np.int8)
    
    # Compute initial cut size efficiently
    # Cut = 0.5 * sum over all i,j of adj[i,j] * (labels[i] != labels[j])
    # = 0.5 * sum over i,j of adj[i,j] * (1 - 2*labels[i]*(1-labels[j]) - 2*(1-labels[i])*labels[j])
    # Simpler: for each i, sum adj[i,:] where labels differ
    diff_mask = (labels[:, None] != labels[None, :])
    current_cut = int((adj * diff_mask).sum() // 2)
    
    # Precompute degree
    degrees = adj.sum(axis=1)
    
    start_time = time.time()
    improved = True
    while improved:
        improved = False
        for v in range(n):
            if time.time() - start_time > time_limit:
                break
            # Gain from flipping v = edges_to_same - edges_to_opposite
            # = (degree[v] - edges_to_opposite) - edges_to_opposite
            # = degree[v] - 2 * edges_to_opposite
            # edges_to_opposite = sum of adj[v, u] where labels[u] != labels[v]
            opposite_count = int(adj[v][labels != labels[v]].sum())
            gain = degrees[v] - 2 * opposite_count
            if gain > 0:
                labels[v] = 1 - labels[v]
                current_cut += gain
                improved = True
        if time.time() - start_time > time_limit:
            break
    
    return labels, current_cut


def partition(adjacency):
    n = len(adjacency)
    # Deadline guard
    global_deadline = time.time() + 110  # 110 seconds out of 120s, leave safety margin
    rng = np.random.default_rng(seed=42)
    
    best_labels = None
    best_cut = -1
    
    # Strategy: Multiple random restarts with local search.
    # Also try a simple alternating start.
    
    num_restarts = 0
    max_restarts = 100  # Upper bound, but time will limit it
    
    while time.time() < global_deadline - 5 and num_restarts < max_restarts:
        # Generate random initial labels
        if num_restarts == 0:
            # First start: alternating
            labels = np.arange(n) % 2
        elif num_restarts == 1:
            # Second start: all zeros (will be improved by local search)
            labels = np.zeros(n, dtype=int)
        elif num_restarts == 2:
            # Third start: all ones
            labels = np.ones(n, dtype=int)
        else:
            labels = rng.integers(0, 2, size=n)
        
        # Run local search with a time limit that shrinks as we approach the deadline
        remaining_time = max(1.0, (global_deadline - 5) - time.time())
        # Give each restart a portion of the remaining time, but at least 0.05s
        ls_time_limit = max(0.05, remaining_time / (num_restarts + 10))
        
        refined_labels, cut = _local_search(labels, adjacency, ls_time_limit, rng)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = refined_labels.copy()
        
        num_restarts += 1
    
    # Fallback if no labels found (should not happen)
    if best_labels is None:
        best_labels = np.zeros(n, dtype=int)
    
    return best_labels
# EVOLVE-BLOCK-END
