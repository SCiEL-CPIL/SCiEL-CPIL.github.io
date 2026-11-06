"""Rewritten program for Maximum Cut using SDP-inspired spectral initialization + efficient local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    adj = np.array(adjacency, dtype=np.int8)
    n = len(adj)
    deadline = time.time() + 130.0  # safety margin before 150s kill

    # Compute spectral initialization using the Laplacian eigenvectors
    deg = adj.sum(axis=1)
    L = np.diag(deg) - adj

    try:
        eigvals, eigvecs = np.linalg.eigh(L)
    except Exception:
        eigvecs = None

    best_labels = None
    best_cut = -1

    # Collect initial partition candidates
    candidates = []

    if eigvecs is not None:
        # Direct eigenvector sign initializations (indices 1..6)
        for idx in range(1, min(7, n)):
            fiedler = eigvecs[:, idx]
            labels = (fiedler > 0).astype(np.int8)
            if labels.sum() == 0 or labels.sum() == n:
                labels[n // 2] = 1 - labels[n // 2]
            candidates.append(labels.copy())

        # SDP random hyperplane initialization: project random hyperplanes
        # onto the subspace spanned by the first k Laplacian eigenvectors
        k_sub = min(6, n - 1)
        V = eigvecs[:, 1:1 + k_sub]  # n x k_sub matrix

        for _ in range(8):
            if time.time() > deadline - 30.0:
                break
            # Random hyperplane in the eigenvector subspace
            r = np.random.randn(k_sub)
            proj = V @ r  # n-vector
            labels = (proj > 0).astype(np.int8)
            if labels.sum() == 0 or labels.sum() == n:
                labels[n // 2] = 1 - labels[n // 2]
            candidates.append(labels)

    # Also add a couple of random initializations for diversity
    for _ in range(3):
        labels = np.random.randint(0, 2, n).astype(np.int8)
        if labels.sum() == 0 or labels.sum() == n:
            labels[n // 2] = 1 - labels[n // 2]
        candidates.append(labels)

    # Refine each candidate with local search
    for labels in candidates:
        if time.time() > deadline - 10.0:
            break
        labels, cut = _local_search(adj, labels, deadline)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()

    # Fallback
    if best_labels is None:
        best_labels = np.arange(n) % 2
        best_cut = _cut_size(adj, best_labels)

    return best_labels


def _local_search(adj, labels, deadline):
    """Run interleaved 1-flip and 2-flip local search until convergence or deadline."""
    n = len(labels)

    while time.time() < deadline:
        changed = False

        # 1-flip pass
        side0 = (labels == 0).astype(np.int8)
        side1 = (labels == 1).astype(np.int8)
        s0_counts = adj @ side0
        s1_counts = adj @ side1

        # Gain if flip v:
        # If label 0: gain = s0[v] - s1[v]
        # If label 1: gain = s1[v] - s0[v]
        flip_gain_1 = np.where(labels == 0, s0_counts - s1_counts, s1_counts - s0_counts)

        # Find the best 1-flip
        max_gain_1 = flip_gain_1.max()
        if max_gain_1 > 0:
            idx = np.argmax(flip_gain_1)
            labels[idx] = 1 - labels[idx]
            changed = True
        else:
            # No 1-flip improvement, try 2-flip
            fg = flip_gain_1
            gain_mat = fg[:, None] + fg[None, :]
            gain_mat += 2 * adj

            # Mask: i < j and labels[i] != labels[j]
            diff_mask = (labels[:, None] != labels[None, :])
            upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
            valid_mask = diff_mask & upper_mask

            masked_gains = np.where(valid_mask, gain_mat, -np.inf)
            max_gain_2 = masked_gains.max()

            if max_gain_2 > 0:
                idx = np.unravel_index(masked_gains.argmax(), (n, n))
                i, j = idx
                labels[i] = 1 - labels[i]
                labels[j] = 1 - labels[j]
                changed = True
            else:
                # No 1-flip or 2-flip improvement
                break

    return labels, _cut_size(adj, labels)


def _cut_size(adj, labels):
    """Compute the number of edges crossing the cut."""
    diff = (labels[:, None] != labels[None, :]).astype(np.int8)
    return int(np.sum(adj * diff) // 2)
# EVOLVE-BLOCK-END
