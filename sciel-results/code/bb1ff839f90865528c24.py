"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adj, labels):
    """Compute the number of edges crossing the cut."""
    diff = (labels[:, None] != labels[None, :]).astype(np.int8)
    return int(np.sum(adj * diff) // 2)


def _flip_gain(adj, labels, v):
    """Compute the gain in cut size if we flip vertex v."""
    # Current contribution of v to cut: neighbors with different label
    # After flip: neighbors with same label become different
    # gain = (num_same_neighbors - num_diff_neighbors) ... wait
    # Let's think carefully.
    # Cut size = 0.5 * sum_{u,v} adj[u,v] * (label[u] != label[v])
    # Contribution of vertex v to the sum (before dividing by 2) is:
    #   sum_u adj[v,u] * (label[v] != label[u])
    # If we flip v, this becomes:
    #   sum_u adj[v,u] * (1 - label[v] != label[u]) = sum_u adj[v,u] * (label[v] == label[u])
    # So the change in the total sum (before /2) is:
    #   new_contrib - old_contrib = sum_u adj[v,u]*(label[v]==label[u]) - sum_u adj[v,u]*(label[v]!=label[u])
    #   = sum_u adj[v,u] - 2 * sum_u adj[v,u]*(label[v]!=label[u])
    #   = degree(v) - 2 * (current crossing count for v)
    # The gain in cut size (after /2) is half of this change.
    # But for comparison purposes, we can just use the change in the sum.
    
    same_count = adj[v][labels == labels[v]].sum()
    diff_count = adj[v][labels != labels[v]].sum()
    # gain in the unhalved sum = same_count - diff_count
    # We flip if this is positive, i.e., same_count > diff_count
    return int(same_count - diff_count)


def _spectral_init(adj, n):
    """Initialize labels using the Fiedler vector (2nd eigenvector of Laplacian)."""
    # Compute Laplacian L = D - A
    degrees = adj.sum(axis=1)
    L = np.diag(degrees) - adj
    # Use eigh for symmetric matrix
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        # Fiedler vector is the eigenvector corresponding to the 2nd smallest eigenvalue
        # Sort eigenvalues (eigh returns them in ascending order)
        fiedler = eigvecs[:, 1]
        # Threshold at 0 to get 0/1 labels
        labels = (fiedler >= 0).astype(int)
        # Ensure both labels are present
        if labels.min() == labels.max():
            # Fallback: alternating
            labels = np.arange(n) % 2
        return labels
    except Exception:
        return np.arange(n) % 2


def _local_search(adj, labels, max_rounds=10, deadline=None):
    """Greedy local search: flip vertices if it improves the cut."""
    n = len(labels)
    best_labels = labels.copy()
    best_cut = _cut_size(adj, best_labels)
    
    for rnd in range(max_rounds):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for v in range(n):
            if deadline is not None and time.time() > deadline:
                break
            gain = _flip_gain(adj, best_labels, v)
            if gain > 0:
                best_labels[v] = 1 - best_labels[v]
                improved = True
                best_cut += gain // 2  # Approximate update; will recompute if needed
        if not improved:
            break
    
    # Recompute exact cut size
    best_cut = _cut_size(adj, best_labels)
    return best_labels, best_cut


def _simulated_annealing(adj, labels, deadline):
    """Simulated annealing to escape local optima."""
    n = len(labels)
    current_labels = labels.copy()
    current_cut = _cut_size(adj, current_labels)
    best_labels = current_labels.copy()
    best_cut = current_cut
    
    # Temperature schedule
    T_start = 10.0
    T_end = 0.1
    cooling = 0.99
    
    T = T_start
    rng = np.random.default_rng(42)
    
    while T > T_end:
        if time.time() > deadline:
            break
        
        # Pick a random vertex to flip
        v = rng.integers(0, n)
        old_label = current_labels[v]
        current_labels[v] = 1 - old_label
        
        new_cut = _cut_size(adj, current_labels)
        delta = new_cut - current_cut
        
        if delta > 0 or rng.random() < np.exp(delta / T):
            current_cut = new_cut
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = current_labels.copy()
        else:
            current_labels[v] = old_label
        
        T *= cooling
    
    return best_labels, best_cut


def partition(adjacency):
    """Find a maximum cut using spectral initialization, local search, and simulated annealing."""
    adj = np.asarray(adjacency, dtype=np.int8)
    n = len(adj)
    
    # Deadline guard: leave 5 seconds safety margin
    start_time = time.time()
    deadline = start_time + 115.0
    
    # 1. Spectral initialization
    labels = _spectral_init(adj, n)
    
    # 2. Local search to polish
    labels, cut = _local_search(adj, labels, max_rounds=20, deadline=deadline)
    
    # 3. Simulated annealing with multiple restarts
    best_labels = labels
    best_cut = cut
    
    # Try a few random restarts if time permits
    rng = np.random.default_rng(123)
    for restart in range(5):
        if time.time() > deadline:
            break
        
        # Random initialization
        rand_labels = rng.integers(0, 2, size=n)
        
        # Local search
        ls_labels, ls_cut = _local_search(adj, rand_labels, max_rounds=10, deadline=deadline)
        
        if ls_cut > best_cut:
            best_cut = ls_cut
            best_labels = ls_labels
        
        # Simulated annealing from the best local optimum
        if time.time() < deadline - 10:
            sa_labels, sa_cut = _simulated_annealing(adj, best_labels, deadline)
            if sa_cut > best_cut:
                best_cut = sa_cut
                best_labels = sa_labels
    
    return best_labels
# EVOLVE-BLOCK-END
