"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    adj = np.array(adjacency, dtype=np.int8)
    n = len(adj)
    deadline = time.time() + 145.0  # safety margin before 150s kill

    # Compute spectral initialization using the Fiedler vector (second eigenvector)
    # For max-cut, we want to maximize the cut, which relates to minimizing the
    # quadratic form x^T L x where L is the Laplacian. The Fiedler vector gives
    # a good initial partition.
    deg = adj.sum(axis=1)
    L = np.diag(deg) - adj
    # Use numpy's eigensolver for the Laplacian
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        # Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
        fiedler = eigvecs[:, 1]
        labels = (fiedler > 0).astype(np.int8)
    except Exception:
        # Fallback to alternating start
        labels = np.arange(n) % 2

    best_labels = labels.copy()
    best_cut = _cut_size(adj, labels)

    # 1-flip local search (vectorized)
    for _ in range(5):
        changed = False
        for v in range(n):
            if time.time() > deadline:
                break
            same = adj[v][labels == labels[v]].sum()
            other = adj[v][labels != labels[v]].sum()
            if same > other:
                labels[v] = 1 - labels[v]
                changed = True
        if not changed:
            break
        cut = _cut_size(adj, labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()

    # 2-flip (pairwise swap) local search
    # For pairs (i, j) with different labels, consider flipping both
    # Gain from flipping both i and j simultaneously
    for _ in range(3):
        changed = False
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if labels[i] == labels[j]:
                    continue
                # Compute gain of flipping both i and j
                # Current contribution: edges from i to same-side + edges from j to same-side
                # After flip: edges from i to other-side + edges from j to other-side
                # But we need to account for the edge between i and j if it exists
                # Let's compute the cut change directly
                old_cut = _cut_size(adj, labels)
                labels[i] = 1 - labels[i]
                labels[j] = 1 - labels[j]
                new_cut = _cut_size(adj, labels)
                if new_cut > old_cut:
                    changed = True
                else:
                    # Revert
                    labels[i] = 1 - labels[i]
                    labels[j] = 1 - labels[j]
        if not changed:
            break
        cut = _cut_size(adj, labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()

    return best_labels


def _cut_size(adj, labels):
    """Compute the number of edges crossing the cut."""
    diff = (labels[:, None] != labels[None, :]).astype(np.int8)
    return int(np.sum(adj * diff) // 2)
# EVOLVE-BLOCK-END
