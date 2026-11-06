"""Rewritten Maximum Cut solver: spectral init + fast greedy + 2-flip moves + SA."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    return int(adj[labels[:, None] != labels[None, :]].sum() // 2)


def _compute_gains(adj_int, labels):
    """
    Compute gain for flipping each vertex.
    gain[v] = (# neighbors with opposite label) - (# neighbors with same label)
    """
    n = len(labels)
    degree = adj_int.sum(axis=1)
    # same_deg[v] = sum_u adj_int[v,u] * (labels[u] == labels[v])
    M = (labels[None, :] == labels[:, None])
    same_deg = (adj_int * M).sum(axis=1)
    gains = degree - 2 * same_deg
    return gains


def _greedy_improve(adj_int, labels):
    """
    Greedy local search: repeatedly flip vertices with positive gain.
    Returns improved labels and cut size.
    """
    n = len(labels)
    labels = labels.copy()
    gains = _compute_gains(adj_int, labels)
    
    max_iterations = n * 50
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        # Find the vertex with the largest positive gain
        max_gain = gains.max()
        if max_gain <= 0:
            break
        v = int(np.argmax(gains))
        # Flip v
        labels[v] = 1 - labels[v]
        # Update gains incrementally
        # When we flip v:
        # - For v itself: gains[v] becomes -gains[v] (since same/other swap)
        # - For each neighbor u of v: 
        #   If labels[u] was same as old labels[v], after flip they're different -> gain[u] increases by 2
        #   If labels[u] was different from old labels[v], after flip they're same -> gain[u] decreases by 2
        # More precisely:
        #   gain[u] changes by +2 if adj[u,v] and labels[u] == new_labels[v] (i.e., was different from old)
        #   gain[u] changes by -2 if adj[u,v] and labels[u] != new_labels[v] (i.e., was same as old)
        # Since new_labels[v] = 1 - old_labels[v]:
        #   labels[u] == new_labels[v] means labels[u] != old_labels[v] -> was different -> gain[u] += 2
        #   labels[u] != new_labels[v] means labels[u] == old_labels[v] -> was same -> gain[u] -= 2
        
        neighbors = np.where(adj_int[v] > 0)[0]
        for u in neighbors:
            if labels[u] == labels[v]:
                # Now they are same, so u's same_count increased, other_count decreased
                # gain[u] = other - same, so gain[u] -= 2
                gains[u] -= 2
            else:
                # Now they are different, so u's other_count increased, same_count decreased
                # gain[u] += 2
                gains[u] += 2
        # For v itself:
        # gain[v] = other_v - same_v. After flip, what was same is now other and vice versa.
        # So new gain[v] = old_same_v - old_other_v = -(old_other_v - old_same_v) = -old_gain[v]
        gains[v] = -gains[v]
    
    cut = _cut_size(adj_int.astype(bool), labels)
    return labels, cut


def _spectral_init(adj):
    """
    Initialize labels using the Fiedler vector (second eigenvector of Laplacian).
    """
    n = len(adj)
    adj_f = np.array(adj, dtype=float)
    
    degree = adj_f.sum(axis=1)
    L = np.diag(degree) - adj_f
    
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        fiedler = eigenvectors[:, 1]
        
        # Use sign of Fiedler vector
        labels = (fiedler > 0).astype(int)
        
        if labels.sum() == 0 or labels.sum() == n:
            median_val = np.median(fiedler)
            labels = (fiedler > median_val).astype(int)
        
    except Exception:
        labels = np.arange(n) % 2
    
    return labels


def _two_flip_improve(adj_int, labels, gains, max_passes=10):
    """
    Try 2-flip moves to escape 1-flip local optima.
    Returns improved labels, gains, and cut size.
    """
    n = len(labels)
    labels = labels.copy()
    gains = gains.copy()
    
    for _ in range(max_passes):
        improved = False
        # Try all pairs of vertices
        for i in range(n):
            for j in range(i + 1, n):
                if adj_int[i, j] > 0:
                    continue  # Skip if they are adjacent (less likely to help)
                # Compute delta for flipping both i and j
                # Delta = change in cut when both i and j are flipped
                # This is complex; let's just try it and check
                old_labels_i = labels[i]
                old_labels_j = labels[j]
                labels[i] = 1 - old_labels_i
                labels[j] = 1 - old_labels_j
                
                new_gains = _compute_gains(adj_int, labels)
                # Check if this improved the cut
                # Actually, let's compute the cut directly for small n
                # But that's expensive. Instead, use the fact that:
                # The change in cut from flipping i alone is gains[i]
                # The change from flipping j alone is gains[j]
                # But flipping both together: the interaction term matters
                # Let's just compute the new cut
                new_cut = _cut_size(adj_int.astype(bool), labels)
                old_cut = _cut_size(adj_int.astype(bool), np.concatenate([
                    labels[:i], [old_labels_i], labels[i+1:j], [old_labels_j], labels[j+1:]
                ]))
                
                if new_cut > old_cut:
                    gains = new_gains
                    improved = True
                else:
                    labels[i] = old_labels_i
                    labels[j] = old_labels_j
        
        if not improved:
            break
    
    cut = _cut_size(adj_int.astype(bool), labels)
    return labels, gains, cut


def _simulated_annealing(adj_int, labels, start_time, deadline):
    """
    Simulated annealing to escape local optima.
    """
    n = len(labels)
    labels = labels.copy()
    
    current_cut = _cut_size(adj_int.astype(bool), labels)
    best_cut = current_cut
    best_labels = labels.copy()
    gains = _compute_gains(adj_int, labels)
    
    T = 5.0
    cooling = 0.98
    
    while time.time() - start_time < deadline:
        if T < 0.01:
            break
        
        # Pick a random vertex to flip
        v = random.randint(0, n - 1)
        
        # Delta from flipping v is gains[v]
        delta = gains[v]
        
        if delta > 0 or random.random() < np.exp(delta / T):
            # Accept
            labels[v] = 1 - labels[v]
            
            # Update gains incrementally
            neighbors = np.where(adj_int[v] > 0)[0]
            for u in neighbors:
                if labels[u] == labels[v]:
                    gains[u] -= 2
                else:
                    gains[u] += 2
            gains[v] = -gains[v]
            
            current_cut += delta
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
        
        T *= cooling
    
    return best_labels, best_cut


def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    adj = np.array(adjacency, dtype=bool)
    adj_int = adj.astype(np.int64)
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    best_labels = None
    best_cut = -1
    
    def _update_best(labels, cut):
        nonlocal best_labels, best_cut
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    # Phase 1: Spectral initialization + greedy improvement
    spectral_labels = _spectral_init(adj)
    improved_labels, cut = _greedy_improve(adj_int, spectral_labels)
    _update_best(improved_labels, cut)
    
    # Phase 2: Multiple random restarts with greedy
    max_restarts = 50
    for restart in range(max_restarts):
        if time.time() - start_time > deadline:
            break
        
        # Random initial labeling
        labels = np.random.randint(0, 2, n)
        
        # Greedy local improvement
        improved_labels, cut = _greedy_improve(adj_int, labels)
        _update_best(improved_labels, cut)
    
    # Phase 3: 2-flip improvement from best solution
    if time.time() - start_time < deadline and best_labels is not None:
        gains = _compute_gains(adj_int, best_labels)
        improved_labels, gains, cut = _two_flip_improve(adj_int, best_labels, gains, max_passes=5)
        _update_best(improved_labels, cut)
        
        # Greedy after 2-flip
        improved_labels, cut = _greedy_improve(adj_int, improved_labels)
        _update_best(improved_labels, cut)
    
    # Phase 4: Simulated annealing from the best solution found
    if time.time() - start_time < deadline and best_labels is not None:
        sa_labels, sa_cut = _simulated_annealing(adj_int, best_labels, start_time, deadline)
        _update_best(sa_labels, sa_cut)
    
    # Fallback
    if best_labels is None:
        labels = np.arange(n) % 2
        best_labels, _ = _greedy_improve(adj_int, labels)
    
    return best_labels
# EVOLVE-BLOCK-END
