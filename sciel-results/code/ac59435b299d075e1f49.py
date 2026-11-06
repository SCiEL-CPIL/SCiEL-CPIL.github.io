"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _greedy_pass(adj_int, deg, labels):
    """One full greedy pass. Returns (new_labels, num_flips)."""
    # same_counts[v] = number of neighbors u of v with labels[u] == labels[v]
    same_mask = (labels[None, :] == labels[:, None])
    same_counts = adj_int @ same_mask
    other_counts = deg - same_counts
    flip = same_counts > other_counts
    if not flip.any():
        return labels, 0
    new_labels = labels.copy()
    new_labels[flip] = 1 - new_labels[flip]
    return new_labels, int(flip.sum())


def partition(adjacency):
    start_time = time.time()
    DEADLINE = 25.0  # per-call safety margin (well under the 120s limit)

    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)

    adj_int = np.asarray(adjacency, dtype=np.int32)
    deg = adj_int.sum(axis=1).astype(np.int64)

    rng = np.random.default_rng(0)

    best_labels = None
    best_cut = -1

    def cut_size(labels):
        same_mask = (labels[None, :] == labels[:, None])
        same_counts = adj_int @ same_mask
        other_counts = deg - same_counts
        return int(other_counts.sum() // 2)

    # First restart: alternating start (deterministic baseline)
    labels = np.arange(n) % 2
    for _ in range(300):
        labels, flips = _greedy_pass(adj_int, deg, labels)
        if flips == 0:
            break
    c = cut_size(labels)
    best_cut = c
    best_labels = labels.copy()

    # Multiple random restarts within the time budget
    while time.time() - start_time < DEADLINE:
        labels = rng.integers(0, 2, size=n).astype(np.int64)
        for _ in range(300):
            labels, flips = _greedy_pass(adj_int, deg, labels)
            if flips == 0:
                break
        c = cut_size(labels)
        if c > best_cut:
            best_cut = c
            best_labels = labels.copy()

    return best_labels
# EVOLVE-BLOCK-END
