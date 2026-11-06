"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _cut_size(adj, labels):
    """Compute the number of edges crossing the partition."""
    # Edge crosses if labels differ
    # Use matrix multiplication trick: for each edge (i,j), it crosses if labels[i] != labels[j]
    # Equivalent to: sum over all i<j of adj[i,j] * (labels[i] != labels[j])
    # Faster: use the fact that cut = 0.5 * sum_i sum_j adj[i,j] * (labels[i] - labels[j])^2 / 4 * 4
    # Actually simpler: cut = sum of adj[i,j] where labels[i] != labels[j]
    # Vectorized: 
    diff = (labels[:, None] != labels[None, :]).astype(np.float64)
    return 0.5 * np.sum(adj * diff)


def _greedy_improve(adj, labels, max_passes=10):
    """Greedy local improvement: flip vertices if it increases cut size."""
    n = len(labels)
    labels = labels.copy()
    # Precompute degrees
    deg = adj.sum(axis=1)
    
    # Track cut contribution per vertex
    # For vertex v in group g, its contribution to cut is number of neighbors in 1-g
    # Let's maintain: for each v, cut_contrib[v] = number of neighbors with different label
    
    cut_contrib = np.zeros(n, dtype=np.float64)
    for v in range(n):
        cut_contrib[v] = adj[v, labels != labels[v]].sum()
    
    current_cut = np.sum(cut_contrib) / 2.0
    
    for _ in range(max_passes):
        improved = False
        for v in range(n):
            same_count = deg[v] - cut_contrib[v]  # neighbors with same label
            # If we flip v, new cut contribution = same_count (old same become different)
            # Delta in total cut = same_count - cut_contrib[v]
            # But we divide by 2 for the total, so delta_cut = (same_count - cut_contrib[v]) / 2
            # Actually, let's think more carefully.
            # Total cut = 0.5 * sum_v cut_contrib[v]
            # If we flip v: cut_contrib[v] becomes same_count
            # For each neighbor u of v: 
            #   if u was different from v, now u is same -> cut_contrib[u] decreases by 1
            #   if u was same as v, now u is different -> cut_contrib[u] increases by 1
            # So delta_total = 0.5 * [(same_count - cut_contrib[v]) + sum_{u neighbor of v} (new_diff - old_diff)]
            # For u neighbor of v: if old same, new diff: +1; if old diff, new same: -1
            # Number of old same neighbors = same_count
            # Number of old diff neighbors = cut_contrib[v]
            # So sum of changes for neighbors = same_count - cut_contrib[v]
            # Delta_total = 0.5 * [(same_count - cut_contrib[v]) + (same_count - cut_contrib[v])]
            #            = same_count - cut_contrib[v]
            delta = same_count - cut_contrib[v]
            if delta > 0:
                # Flip v
                labels[v] = 1 - labels[v]
                # Update cut_contrib for v and its neighbors
                old_contrib = cut_contrib[v]
                cut_contrib[v] = same_count
                # Update neighbors
                neighbors = np.where(adj[v] > 0)[0]
                for u in neighbors:
                    # u's relationship with v flipped
                    if labels[u] == labels[v]:
                        # Now same, was different
                        cut_contrib[u] -= 1
                    else:
                        # Now different, was same
                        cut_contrib[u] += 1
                improved = True
        if not improved:
            break
    
    return labels


def _spectral_init(adj, n):
    """Spectral initialization using Fiedler vector (second eigenvector of Laplacian)."""
    # Build Laplacian
    deg = adj.sum(axis=1)
    L = sp.diags(deg) - sp.csr_matrix(adj)
    
    # Find second smallest eigenvalue and eigenvector
    # Use eigsh for smallest eigenvalues
    try:
        # We want the second smallest eigenvalue (Fiedler value)
        # eigsh finds k smallest by default with which='SM'
        vals, vecs = spla.eigsh(L, k=2, which='SM')
        # Sort by eigenvalue
        idx = np.argsort(vals)
        fiedler_vec = vecs[:, idx[1]]
    except Exception:
        # Fallback: use first eigenvector or random
        fiedler_vec = np.random.randn(n)
    
    # Threshold the Fiedler vector at 0 (or median) to get binary labels
    threshold = np.median(fiedler_vec)
    labels = (fiedler_vec > threshold).astype(int)
    
    return labels


def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin
    
    adj = np.asarray(adjacency, dtype=np.float64)
    
    best_labels = None
    best_cut = -1
    
    # Spectral initialization
    labels_spec = _spectral_init(adj, n)
    # Refine with greedy
    labels_spec = _greedy_improve(adj, labels_spec, max_passes=10)
    cut_spec = _cut_size(adj, labels_spec)
    
    if cut_spec > best_cut:
        best_cut = cut_spec
        best_labels = labels_spec.copy()
    
    # Also try alternating initialization as in original
    labels_alt = np.arange(n) % 2
    labels_alt = _greedy_improve(adj, labels_alt, max_passes=10)
    cut_alt = _cut_size(adj, labels_alt)
    
    if cut_alt > best_cut:
        best_cut = cut_alt
        best_labels = labels_alt.copy()
    
    # Multi-start with random initializations using remaining time
    rng = np.random.RandomState(42)
    
    while time.time() < deadline:
        # Random initialization
        labels_rand = rng.randint(0, 2, size=n)
        labels_rand = _greedy_improve(adj, labels_rand, max_passes=10)
        cut_rand = _cut_size(adj, labels_rand)
        
        if cut_rand > best_cut:
            best_cut = cut_rand
            best_labels = labels_rand.copy()
        
        # Check time
        if time.time() > deadline - 5:
            break
    
    # Final polish: one more greedy pass with more iterations on the best
    if best_labels is not None and time.time() < deadline - 2:
        best_labels = _greedy_improve(adj, best_labels, max_passes=20)
    
    return best_labels
# EVOLVE-BLOCK-END
