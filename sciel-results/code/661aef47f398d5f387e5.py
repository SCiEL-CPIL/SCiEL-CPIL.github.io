"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    # An edge (i,j) crosses if labels[i] != labels[j]
    # Using matrix: for each edge, check if endpoints differ
    cross_mask = (labels[:, None] != labels[None, :])
    return int(np.sum(adj * cross_mask) // 2)


def _spectral_init(adj, n, rng):
    """Spectral (Fiedler vector) initialization for max-cut."""
    deg = adj.sum(axis=1)
    L = np.diag(deg) - adj
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        # Fiedler vector is the second smallest eigenvector
        fiedler = eigvecs[:, 1]
        # Threshold at median for a balanced partition
        med = np.median(fiedler)
        labels = (fiedler > med).astype(np.int8)
    except Exception:
        # Fallback: random balanced partition
        labels = (rng.random(n) > 0.5).astype(np.int8)
    return labels


def _single_flip_ls(adj, labels, deg, max_passes=50):
    """Greedy single-flip local search. Returns improved labels and cut size."""
    n = len(labels)
    labels = labels.copy()
    for _ in range(max_passes):
        improved = False
        # Compute crossing count for each vertex
        # cross[v] = number of neighbors of v on the opposite side
        opp = (labels[:, None] != labels[None, :])  # n x n boolean
        cross = (adj * opp).sum(axis=1)  # for each v, count opposite neighbors
        gain = deg - 2 * cross  # gain of flipping v
        # Find vertices with positive gain
        to_flip = np.where(gain > 0)[0]
        if len(to_flip) == 0:
            break
        for v in to_flip:
            labels[v] = 1 - labels[v]
            improved = True
        if not improved:
            break
    return labels


def _pair_flip_ls(adj, labels, deg, max_passes=20):
    """Pair-flip (2-opt) local search to escape single-flip optima."""
    n = len(labels)
    labels = labels.copy()
    for _ in range(max_passes):
        improved = False
        # Compute current crossing counts
        opp = (labels[:, None] != labels[None, :])
        cross = (adj * opp).sum(axis=1)
        gain = deg - 2 * cross  # single-flip gain for each vertex
        
        # For pair flips, we need to consider the interaction
        # Flipping u and v together: 
        # If u and v are on the same side, the edge (u,v) if it exists doesn't change crossing status
        # If u and v are on opposite sides, the edge (u,v) if it exists changes crossing status
        # Let's compute the combined gain for all pairs
        # combined_gain[u,v] = gain[u] + gain[v] + correction for edge (u,v)
        # The correction: if u and v are connected and on the same side, 
        #   flipping both: edge (u,v) was non-crossing, after flip still non-crossing (both flip) -> no change
        #   Wait, if both flip, same side -> still same side, edge status unchanged.
        # If u and v are connected and on opposite sides:
        #   edge was crossing, after both flip still opposite -> still crossing, no change.
        # Actually, when both flip, the relative relationship between u and v is preserved.
        # But the relationship between u and others, and v and others, changes.
        # The single-flip gain for u already accounts for u's neighbors.
        # When we flip both u and v, the gain is gain[u] + gain[v] + adjustment.
        # The adjustment comes from the fact that when we flip u, v's label changes too,
        # so the edge (u,v) contribution is counted differently.
        
        # Let me think more carefully.
        # Current cut contribution involving u: for each neighbor w of u, edge (u,w) crosses iff labels[u] != labels[w].
        # After flipping u only: edge (u,w) crosses iff (1-labels[u]) != labels[w] = labels[u] == labels[w].
        # So flipping u changes the cut by: (non-crossing edges from u) - (crossing edges from u) = (deg[u] - cross[u]) - cross[u] = deg[u] - 2*cross[u] = gain[u].
        
        # When flipping both u and v:
        # For edges from u to w (w != v): same as flipping u alone, contribution change = gain from u's perspective, but w's label is unchanged.
        # For edges from v to w (w != u): same as flipping v alone.
        # For edge (u,v) if it exists: 
        #   Before: crosses iff labels[u] != labels[v]
        #   After: crosses iff (1-labels[u]) != (1-labels[v]) = labels[u] == labels[v]
        #   So the crossing status of (u,v) flips! It was crossing iff labels differ, now crosses iff labels same.
        #   Change in cut from edge (u,v): 
        #     If labels[u] != labels[v] (was crossing): after flip, labels same, not crossing. Change = -1.
        #     If labels[u] == labels[v] (was not crossing): after flip, labels differ, crossing. Change = +1.
        #   So change = +1 if labels[u]==labels[v], -1 if labels[u]!=labels[v].
        #   This equals: +1 if same side, -1 if opposite side = (1 if labels[u]==labels[v] else -1) = 1 - 2*(labels[u]!=labels[v])
        
        # So combined_gain[u,v] = gain[u] + gain[v] + adj_uv
        # where adj_uv = +1 if labels[u]==labels[v] else -1, but ONLY if edge (u,v) exists.
        # If no edge (u,v), adj_uv = 0.
        
        # Let's compute this for all pairs
        same_side = (labels[:, None] == labels[None, :])  # n x n
        # adj_uv: +1 if same side and edge exists, -1 if opposite side and edge exists, 0 if no edge
        adj_uv = np.where(same_side, adj, -adj)  # +adj if same side, -adj if opposite
        # combined_gain matrix
        combined_gain = gain[:, None] + gain[None, :] + adj_uv
        # Only consider pairs u < v
        # Find pairs with positive combined gain
        iu, iv = np.where(np.triu(combined_gain > 0, k=1))
        if len(iu) == 0:
            break
        # Apply the best pair flip (or all improving ones)
        # To be safe, apply the single best pair
        best_idx = np.argmax(combined_gain[iu, iv])
        u = iu[best_idx]
        v = iv[best_idx]
        labels[u] = 1 - labels[u]
        labels[v] = 1 - labels[v]
        improved = True
        if not improved:
            break
    return labels


def partition(adjacency):
    adj = np.array(adjacency, dtype=np.int8)
    n = len(adj)
    deg = adj.sum(axis=1).astype(np.float64)
    deadline = time.time() + 145.0  # safety margin before 150s kill
    
    best_labels = None
    best_cut = -1
    
    rng = np.random.default_rng(42)
    
    # Generate multiple initializations: spectral + random perturbations
    inits = []
    
    # Spectral init
    spec_labels = _spectral_init(adj, n, rng)
    inits.append(spec_labels)
    
    # Add several random balanced inits for diversity
    for _ in range(5):
        labels = (rng.random(n) > 0.5).astype(np.int8)
        inits.append(labels)
    
    # Also add perturbed spectral inits
    for _ in range(3):
        perturbed = spec_labels.copy()
        idx = rng.choice(n, size=n // 4, replace=False)
        perturbed[idx] = 1 - perturbed[idx]
        inits.append(perturbed)
    
    for init_labels in inits:
        if time.time() > deadline - 5.0:
            break
        
        # Single-flip local search
        labels = _single_flip_ls(adj, init_labels, deg, max_passes=50)
        
        # Pair-flip local search
        labels = _pair_flip_ls(adj, labels, deg, max_passes=20)
        
        # Final single-flip polish
        labels = _single_flip_ls(adj, labels, deg, max_passes=20)
        
        cut = _cut_size(adj, labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    if best_labels is None:
        # Fallback: alternating
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
