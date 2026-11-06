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

    # Generate multiple initializations from spectral vectors and random hyperplanes
    initializations = []
    if eigvecs is not None:
        # Use several low-frequency eigenvectors as starting points
        for idx in range(1, min(8, n)):
            fiedler = eigvecs[:, idx]
            labels = (fiedler > 0).astype(np.int8)
            if labels.sum() != 0 and labels.sum() != n:
                initializations.append(labels)
            else:
                labels[n // 2] = 1 - labels[n // 2]
                initializations.append(labels)

        # SDP-inspired: project random hyperplanes onto the subspace spanned by top eigenvectors
        # Use the first k eigenvectors to form a basis for the relaxation
        k_sdp = min(6, n)
        V = eigvecs[:, 1:k_sdp + 1]  # Shape: (n, k_sdp)

        rng = np.random.default_rng(42)
        num_hyperplanes = 8
        for h in range(num_hyperplanes):
            if time.time() > deadline - 40.0:
                break
            u = rng.standard_normal(k_sdp)
            u = u / np.linalg.norm(u)
            projections = V @ u  # Shape: (n,)
            labels_hp = (projections > 0).astype(np.int8)
            if labels_hp.sum() != 0 and labels_hp.sum() != n:
                initializations.append(labels_hp)

    # Add a few random initializations for diversity
    rng2 = np.random.default_rng(123)
    for _ in range(3):
        labels = rng2.integers(0, 2, n).astype(np.int8)
        if labels.sum() != 0 and labels.sum() != n:
            initializations.append(labels)

    # If no initializations, use alternating
    if not initializations:
        initializations.append((np.arange(n) % 2).astype(np.int8))

    # Run local search from each initialization
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
        flip_gain = np.where(labels == 0, s0_counts - s1_counts, s1_counts - s0_counts)

        # Find the best 1-flip
        max_gain_1 = flip_gain.max()
        if max_gain_1 > 0:
            idx = np.argmax(flip_gain)
            labels[idx] = 1 - labels[idx]
            changed = True
        else:
            # No 1-flip improvement, try 2-flip
            # Gain(i, j) for i<j with labels[i] != labels[j]:
            #   gain = flip_gain[i] + flip_gain[j] + 2 * adj[i,j]
            #         - (if labels[i]==labels[j] then 0, but we mask those out)
            fg = flip_gain
            gain_mat = fg[:, None] + fg[None, :]
            gain_mat += 2 * adj

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
