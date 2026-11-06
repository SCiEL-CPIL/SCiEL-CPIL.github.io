# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(labels, adj):
    """Compute cut size: number of edges with endpoints in different partitions."""
    # adj is a bool or int matrix; count edges where labels differ
    # Use matrix multiplication trick: for 0/1 labels, cut = sum(adj * (labels_i != labels_j))
    # More efficiently: cut = (sum of degrees into other partition) / 2
    # Or: cut = 0.5 * (labels @ adj @ (1 - labels) + (1 - labels) @ adj @ labels)
    # Actually simpler: cut = labels @ adj @ (1 - labels) ... no that's not right for undirected.
    
    # For undirected graph stored symmetrically:
    # An edge (i,j) is in the cut if labels[i] != labels[j].
    # cut = sum_{i<j, adj[i,j]} [labels[i] != labels[j]]
    # = 0.5 * sum_{i,j} adj[i,j] * [labels[i] != labels[j]]
    # [labels[i] != labels[j]] = 1 - labels[i]*labels[j] - (1-labels[i])*(1-labels[j])
    # = 1 - 2*labels[i]*(1-labels[j]) - ... hmm let me just do it directly.
    
    # Vectorized: 
    diff = (labels[:, None] != labels[None, :])
    return int((adj * diff).sum() // 2)  # divide by 2 since symmetric


def _local_search(labels, adj, adj_int, degrees, time_limit):
    """
    Local search: flip individual vertices to improve cut.
    Returns best labels found.
    """
    n = len(labels)
    labels = labels.copy()
    best_cut = _cut_size(labels, adj)
    best_labels = labels.copy()
    
    deadline = time.time() + time_limit
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        for v in range(n):
            if time.time() >= deadline:
                break
            current_label = labels[v]
            # Gain from flipping v:
            # Currently, edges from v to same side are "wasted" (not in cut)
            # After flip, those become cut edges, and edges to other side become non-cut
            # gain = (# neighbors on other side) - (# neighbors on same side)
            #      = 2 * (# neighbors on other side) - degree[v]
            same_count = np.sum(adj_int[v, labels == current_label])
            other_count = degrees[v] - same_count
            gain = 2 * other_count - degrees[v]
            if gain > 0:
                labels[v] = 1 - current_label
                best_cut += gain
                improved = True
        if best_cut > _cut_size(best_labels, adj):
            best_labels = labels.copy()
    
    # Verify and return
    final_cut = _cut_size(best_labels, adj)
    return best_labels, final_cut


def partition(adjacency):
    """
    Partition vertices into two groups to maximize the number of crossing edges.
    Uses multiple random restarts with local search.
    """
    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=np.int8)
    degrees = adj.sum(axis=1).astype(np.int32)
    
    start_time = time.time()
    total_budget = 110.0  # leave 10s margin for 120s limit (or 150s kill)
    
    rng = np.random.RandomState(42)
    
    best_labels = None
    best_cut = -1
    
    # Try several initializations
    num_restarts = 0
    while time.time() - start_time < total_budget:
        # Choose initialization strategy
        num_restarts += 1
        if num_restarts % 3 == 1:
            # Random
            labels = rng.randint(0, 2, size=n)
        elif num_restarts % 3 == 2:
            # Alternating
            labels = np.arange(n) % 2
        else:
            # All zeros (will be flipped by local search)
            labels = np.zeros(n, dtype=np.int8)
        
        time_left = total_budget - (time.time() - start_time)
        if time_left < 1.0:
            break
        
        # Allocate time for local search, leaving some for next restart
        # Use most of remaining time for the last few restarts
        if num_restarts < 5:
            ls_time = min(20.0, time_left * 0.4)
        else:
            ls_time = time_left * 0.5
        
        labels, cut = _local_search(labels, adj, adj, degrees, ls_time)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
        
        # If we've used most of the budget, stop
        if time.time() - start_time > total_budget * 0.85:
            break
    
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
