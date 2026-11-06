"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def partition(adjacency):
    adj = np.array(adjacency, dtype=np.int8)
    n = len(adj)
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    # Compute spectral initialization using the Fiedler vector (second eigenvector)
    deg = adj.sum(axis=1)
    L = np.diag(deg) - adj
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
    except Exception:
        eigvecs = None

    best_labels = None
    best_cut = -1

    # Generate diverse initializations
    initializations = []
    
    if eigvecs is not None:
        # Use top eigenvectors for Goemans-Williamson style random hyperplane rounding
        # Take top k eigenvectors (excluding the constant first one)
        k = min(8, n - 1)
        top_eigvecs = eigvecs[:, 1:1+k]  # shape (n, k)
        
        # For each of several random directions, project onto that direction and threshold
        num_hyperplanes = 10
        for _ in range(num_hyperplanes):
            if time.time() > deadline - 10.0:
                break
            # Random unit vector in R^k
            v = np.random.randn(k)
            v /= np.linalg.norm(v)
            # Project eigenvectors onto this direction
            projection = top_eigvecs @ v  # shape (n,)
            labels = (projection > 0).astype(np.int8)
            if labels.sum() == 0 or labels.sum() == n:
                labels[n // 2] = 1 - labels[n // 2]
            initializations.append(labels.copy())
        
        # Also add the pure Fiedler vector thresholding
        fiedler = eigvecs[:, 1]
        labels_fiedler = (fiedler > 0).astype(np.int8)
        if labels_fiedler.sum() == 0 or labels_fiedler.sum() == n:
            labels_fiedler[n // 2] = 1 - labels_fiedler[n // 2]
        initializations.append(labels_fiedler)
    else:
        # Fallback: random initializations
        for _ in range(5):
            labels = np.random.randint(0, 2, n).astype(np.int8)
            initializations.append(labels)

    # Run local search on each initialization
    for labels_init in initializations:
        if time.time() > deadline - 5.0:
            break
        labels = labels_init.copy()
        labels, cut = _local_search(adj, labels, deadline)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()

    # If no valid labels yet, use alternating
    if best_labels is None:
        best_labels = np.arange(n) % 2
        best_cut = _cut_size(adj, best_labels)

    # Final polish pass: more aggressive local search on the best solution
    if time.time() < deadline - 2.0:
        best_labels, best_cut = _polish_pass(adj, best_labels, deadline)

    return best_labels


def _local_search(adj, labels, deadline):
    """Run 1-flip and efficient 2-flip local search until convergence or deadline."""
    n = len(adj)
    
    # 1-flip local search with batch updates
    for _ in range(30):
        if time.time() > deadline:
            break
        changed = False
        # Compute flip gains for all vertices at once
        side0_count = adj @ (labels == 0).astype(np.int8)
        side1_count = adj @ (labels == 1).astype(np.int8)
        flip_gain = (2 * labels.astype(np.int8) - 1) * (side1_count - side0_count)
        
        # Find all vertices with positive gain
        to_flip = np.where(flip_gain > 0)[0]
        if len(to_flip) == 0:
            break
        # Apply all flips simultaneously (this is a valid parallel update)
        labels[to_flip] = 1 - labels[to_flip]
        changed = True
    
    # 2-flip local search with batch updates
    for _ in range(50):
        if time.time() > deadline:
            break
        # Precompute side counts
        side0_count = adj @ (labels == 0).astype(np.int8)
        side1_count = adj @ (labels == 1).astype(np.int8)
        
        # Compute flip gain for each vertex
        flip_gain = (2 * labels.astype(np.int8) - 1) * (side1_count - side0_count)
        
        # For 2-flip: gain_matrix[i,j] = flip_gain[i] + flip_gain[j] if labels[i] != labels[j]
        gain_matrix = flip_gain[:, None] + flip_gain[None, :]
        
        # Mask: only pairs with different labels and i < j
        diff_mask = (labels[:, None] != labels[None, :])
        upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        valid_mask = diff_mask & upper_mask
        
        # Find all pairs with positive gain
        masked_gains = np.where(valid_mask, gain_matrix, -np.inf)
        positive_pairs = np.where(masked_gains > 0)
        
        if len(positive_pairs[0]) == 0:
            break
        
        # Apply all positive-gain pairs simultaneously
        # But we need to be careful: if multiple pairs share a vertex, applying them all
        # may not be valid. For simplicity, apply the best pair first, then recompute.
        # Actually, for efficiency, let's apply all non-conflicting pairs.
        # A simpler approach: apply the top few pairs that don't conflict.
        
        # Get the indices of positive-gain pairs
        i_indices, j_indices = positive_pairs
        gains = masked_gains[i_indices, j_indices]
        
        # Sort by gain descending
        order = np.argsort(gains)[::-1]
        
        # Apply non-conflicting pairs
        used = np.zeros(n, dtype=bool)
        applied = 0
        for idx in order:
            if time.time() > deadline:
                break
            i, j = i_indices[idx], j_indices[idx]
            if not used[i] and not used[j]:
                labels[i] = 1 - labels[i]
                labels[j] = 1 - labels[j]
                used[i] = True
                used[j] = True
                applied += 1
                # Limit to avoid too many changes at once
                if applied >= 10:
                    break
        
        if applied == 0:
            break
    
    return labels, _cut_size(adj, labels)


def _polish_pass(adj, labels, deadline):
    """High-precision polish: more iterations of 1-flip and 2-flip, plus 3-flip moves."""
    n = len(adj)
    
    # Extended 1-flip polish
    for _ in range(50):
        if time.time() > deadline:
            break
        side0_count = adj @ (labels == 0).astype(np.int8)
        side1_count = adj @ (labels == 1).astype(np.int8)
        flip_gain = (2 * labels.astype(np.int8) - 1) * (side1_count - side0_count)
        to_flip = np.where(flip_gain > 0)[0]
        if len(to_flip) == 0:
            break
        labels[to_flip] = 1 - labels[to_flip]
    
    # Extended 2-flip polish
    for _ in range(100):
        if time.time() > deadline:
            break
        side0_count = adj @ (labels == 0).astype(np.int8)
        side1_count = adj @ (labels == 1).astype(np.int8)
        flip_gain = (2 * labels.astype(np.int8) - 1) * (side1_count - side0_count)
        gain_matrix = flip_gain[:, None] + flip_gain[None, :]
        diff_mask = (labels[:, None] != labels[None, :])
        upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        valid_mask = diff_mask & upper_mask
        masked_gains = np.where(valid_mask, gain_matrix, -np.inf)
        max_gain = masked_gains.max()
        if max_gain <= 0:
            break
        idx = np.unravel_index(masked_gains.argmax(), (n, n))
        i, j = idx
        labels[i] = 1 - labels[i]
        labels[j] = 1 - labels[j]
    
    # 3-flip moves for escaping local optima
    for _ in range(30):
        if time.time() > deadline:
            break
        # Try random 3-flip moves if they improve the cut
        current_cut = _cut_size(adj, labels)
        improved = False
        for _ in range(5):  # Try a few random 3-flip moves
            if time.time() > deadline:
                break
            # Pick 3 vertices with mixed labels
            if np.sum(labels == 0) < 2 or np.sum(labels == 1) < 1:
                break
            zero_verts = np.where(labels == 0)[0]
            one_verts = np.where(labels == 1)[0]
            # Pick 2 from one side, 1 from the other (or vice versa)
            if len(zero_verts) >= 2 and len(one_verts) >= 1:
                i, j = np.random.choice(zero_verts, size=2, replace=False)
                k = np.random.choice(one_verts)
            elif len(one_verts) >= 2 and len(zero_verts) >= 1:
                i, j = np.random.choice(one_verts, size=2, replace=False)
                k = np.random.choice(zero_verts)
            else:
                break
            
            # Try flipping i, j, k
            old_labels = labels.copy()
            labels[i] = 1 - labels[i]
            labels[j] = 1 - labels[j]
            labels[k] = 1 - labels[k]
            new_cut = _cut_size(adj, labels)
            if new_cut > current_cut:
                current_cut = new_cut
                improved = True
            else:
                labels = old_labels
        
        if not improved:
            break
    
    return labels, _cut_size(adj, labels)


def _cut_size(adj, labels):
    """Compute the number of edges crossing the cut."""
    diff = (labels[:, None] != labels[None, :]).astype(np.int8)
    return int(np.sum(adj * diff) // 2)
# EVOLVE-BLOCK-END
