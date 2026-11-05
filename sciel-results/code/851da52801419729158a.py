# EVOLVE-BLOCK-START
import time
import numpy as np


def _cut_size(labels, gain):
    # cut = (1/2) * sum_v gain[v]  for the 2-partition defined by gain
    return 0.5 * float(gain.sum())


def _local_search(adj, labels, n, m, deadline):
    # labels: int array of 0/1
    # gain[v] = (# neighbors in opposite part) - (# neighbors in same part)
    # Flipping v changes the cut by gain[v].
    labels = labels.copy()
    gain = (adj @ (1 - 2 * labels)).astype(np.int64)  # opp - same per vertex
    cut = _cut_size(labels, gain)

    no_improve = 0
    while time.time() < deadline:
        improved = False
        for v in range(n):
            if time.time() >= deadline:
                break
            g = gain[v]
            if g > 0:  # flipping v increases the cut
                labels[v] = 1 - labels[v]
                cut += g
                improved = True
                # update gain for v and its neighbors
                nb = np.flatnonzero(adj[v])
                gain[v] = -g
                gain[nb] -= 2
        if not improved:
            no_improve += 1
            if no_improve >= 2:
                break
            # restart the sweep from the other side occasionally
        else:
            no_improve = 0
    return labels, cut


def partition(adjacency):
    t0 = time.time()
    HARD_DEADLINE = t0 + 140.0  # safety margin below the 150s kill
    deadline = HARD_DEADLINE

    A = np.asarray(adjacency, dtype=bool)
    n = A.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    m = int(A.sum()) // 2

    best_labels = None
    best_cut = -1

    # Seed: deterministic alternating start first (cheap, reliable).
    seed_labels = (np.arange(n) % 2).astype(np.int64)
    lab, cut = _local_search(A, seed_labels, n, m, deadline)
    best_labels, best_cut = lab, cut

    rng = np.random.default_rng(12345)
    # Random restarts while we have time. Each restart is a full local search.
    while time.time() < deadline - 1.0:
        lab0 = rng.integers(0, 2, size=n).astype(np.int64)
        lab, cut = _local_search(A, lab0, n, m, deadline)
        if cut > best_cut:
            best_cut = cut
            best_labels = lab

    if best_labels is None:
        best_labels = seed_labels
    return best_labels.astype(np.int64)
# EVOLVE-BLOCK-END
