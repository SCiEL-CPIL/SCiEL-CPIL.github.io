import numpy as np
import time

def _spectral_init(adj):
    """
    Initialize labels using the second smallest eigenvector of the Laplacian.
    """
    n = len(adj)
    # Laplacian L = D - A
    degrees = np.sum(adj, axis=1)
    L = np.diag(degrees) - adj
    
    # Compute eigenvalues and eigenvectors
    # We want the Fiedler vector (eigenvector corresponding to the second smallest eigenvalue)
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        # Sort by eigenvalue
        idx = np.argsort(eigvals)
        eigvecs = eigvecs[:, idx]
        # The first eigenvector is usually constant (for connected graphs).
        # The second is the Fiedler vector.
        # Sometimes index 0 or 1 depending on connectivity. 
        # For Max Cut, we often use the eigenvector of A or L.
        # Let's try the second smallest eigenvector of L.
        fiedler = eigvecs[:, 1]
        
        # Partition based on sign relative to mean
        mean_val = np.mean(fiedler)
        labels = (fiedler > mean_val).astype(int)
        
        # Ensure not all 0 or all 1
        if np.all(labels == 0) or np.all(labels == 1):
             labels = np.arange(n) % 2
             
    except Exception:
        labels = np.random.randint(0, 2, n)
        
    return labels

def _fast_local_search(labels, adj):
    """
    Perform local search using single flips with incremental gain updates.
    Returns the improved labels and the cut size.
    """
    n = len(labels)
    labels = labels.copy()
    
    # gain[i] = (number of neighbors on opposite side) - (number of neighbors on same side)
    # If gain[i] > 0, flipping i increases the cut by gain[i].
    
    # Initialize gain array
    gain = np.zeros(n)
    for i in range(n):
        # Sum of adjacency to opposite side
        opp = np.sum(adj[i, labels != labels[i]])
        # Sum of adjacency to same side
        same = np.sum(adj[i, labels == labels[i]])
        gain[i] = opp - same
        
    # Compute initial cut size
    cut_size = np.sum(adj * (labels[:, None] != labels[None, :]))
    
    improved = True
    # Limit iterations to prevent infinite loops in case of ties (shouldn't happen with > 0 check)
    max_iters = 1000
    iters = 0
    
    while improved and iters < max_iters:
        improved = False
        iters += 1
        
        # Find the vertex with the maximum positive gain
        max_gain_idx = np.argmax(gain)
        max_gain_val = gain[max_gain_idx]
        
        if max_gain_val <= 0:
            break
            
        # Flip the vertex
        v = max_gain_idx
        old_label = labels[v]
        labels[v] = 1 - old_label
        
        # Update cut size
        cut_size += max_gain_val
        
        # Update gains for neighbors of v
        # For each neighbor u of v:
        # If u and v were on same side (before flip), they are now on opposite.
        #   gain[u] changes from (opp_u - same_u) to (opp_u + 1 - (same_u - 1)) = gain[u] + 2
        # If u and v were on opposite side (before flip), they are now on same.
        #   gain[u] changes from (opp_u - same_u) to (opp_u - 1 - (same_u + 1)) = gain[u] - 2
        
        neighbors = np.where(adj[v] == 1)[0]
        for u in neighbors:
            if labels[u] == labels[v]:
                # They are now on the same side. Before flip, they were on opposite.
                # Wait, logic check:
                # Before flip: labels[u] != old_label.
                # After flip: labels[u] == 1 - old_label = labels[v].
                # So before, they were opposite. Now same.
                # gain[u] decreases by 2.
                gain[u] -= 2
            else:
                # Before flip: labels[u] == old_label.
                # After flip: labels[u] != 1 - old_label = labels[v].
                # So before, they were same. Now opposite.
                # gain[u] increases by 2.
                gain[u] += 2
        
        # Note: gain[v] itself is not needed for the next step's selection if we just re-calculate or if we know it's now negative?
        # Actually, gain[v] is now -max_gain_val (approximately, ignoring interactions with other flips which haven't happened yet).
        # Since we only flip one at a time, gain[v] becomes -old_gain.
        gain[v] = -max_gain_val

    return labels, cut_size

def _cut_size(labels, adjacency):
    """Compute the number of edges crossing the partition efficiently."""
    diff_matrix = (labels[:, None] != labels[None, :])
    return np.sum(adjacency * diff_matrix)

def partition(adjacency):
    n = len(adjacency)
    deadline = time.time() + 145  # Safety margin before 150s limit
    
    best_labels = None
    best_cut = -1
    
    # 1. Spectral initialization
    labels_spec = _spectral_init(adjacency)
    labels, cut = _fast_local_search(labels_spec, adjacency)
    best_labels = labels.copy()
    best_cut = cut
    
    # 2. Iterated Local Search with random perturbations
    # We will run ILS until time runs out.
    # Each iteration: Perturb best_labels, run local search, keep best.
    
    # Also try a few random starts just in case spectral is bad
    for _ in range(5):
        if time.time() > deadline:
            break
        labels_rand = np.random.randint(0, 2, n)
        labels, cut = _fast_local_search(labels_rand, adjacency)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
            
    # ILS Loop
    # Perturbation strength: flip ~10% of vertices
    perturb_count = max(1, n // 10)
    
    while time.time() < deadline:
        # Start from current best
        current_labels = best_labels.copy()
        
        # Perturb
        if perturb_count > 0:
            flip_indices = np.random.choice(n, perturb_count, replace=False)
            current_labels[flip_indices] = 1 - current_labels[flip_indices]
            
        # Local Search
        labels, cut = _fast_local_search(current_labels, adjacency)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
            
            # Optionally adjust perturb_count or strategy
            # If we found a better one, we can keep perturbing from it.
            
    if best_labels is None:
        best_labels = np.arange(n) % 2
        
    return best_labels
