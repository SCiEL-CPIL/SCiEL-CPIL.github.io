"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
'''Neighbour joining on Jukes-Cantor corrected distances, vectorized with NumPy.'''
import math
import time
import numpy as np


def _jc_correct(p):
    """Jukes-Cantor correction: d = -3/4 * ln(1 - 4p/3)."""
    p = np.clip(p, 0.0, 0.7499)
    return -0.75 * np.log(1.0 - 4.0 * p / 3.0)


def _build_distance_matrix(sequences, labels):
    """Build pairwise JC-corrected distance matrix using vectorized NumPy ops."""
    n = len(labels)
    seq_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    L = len(sequences[labels[0]])
    arr = np.zeros((n, L), dtype=np.int8)
    for i, lab in enumerate(labels):
        s = sequences[lab]
        for j, ch in enumerate(s):
            arr[i, j] = seq_map[ch]
    # Hamming distance matrix
    diff = arr[:, None, :] != arr[None, :, :]
    hamming = diff.sum(axis=2).astype(np.float64)
    p = hamming / L
    d = _jc_correct(p)
    np.fill_diagonal(d, 0.0)
    return d


def reconstruct(sequences):
    """Reconstruct unrooted tree topology from DNA sequences using Neighbor-Joining."""
    start_time = time.time()
    deadline = start_time + 170.0

    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)

    if n <= 2:
        if n == 0:
            return ""
        if n == 1:
            return labels[0] + ";"
        return "(" + ",".join(labels) + ");"

    # Build initial distance matrix
    d_init = _build_distance_matrix(sequences, labels)

    # Pre-allocate larger matrix for NJ (up to 2n-1 nodes)
    max_nodes = 2 * n
    d_work = np.zeros((max_nodes, max_nodes), dtype=np.float64)
    d_work[:n, :n] = d_init

    trees = {}
    for i, lab in enumerate(labels):
        trees[i] = lab

    active = list(range(n))
    next_id = n

    while len(active) > 3:
        if time.time() > deadline:
            break

        m = len(active)
        # Compute row sums for active nodes
        row_sums = {}
        for ai in active:
            s = 0.0
            for aj in active:
                if aj != ai:
                    s += d_work[ai, aj]
            row_sums[ai] = s

        # Find pair with minimum Q criterion
        best_q = float('inf')
        best_a = best_b = -1
        for i in range(m):
            for j in range(i + 1, m):
                a, b = active[i], active[j]
                q = (m - 2) * d_work[a, b] - row_sums[a] - row_sums[b]
                if q < best_q:
                    best_q = q
                    best_a, best_b = a, b

        a, b = best_a, best_b

        # Compute branch lengths
        m_minus_2 = m - 2
        x = (row_sums[a] - row_sums[b]) / (2.0 * m_minus_2) + d_work[a, b] / 2.0
        y = d_work[a, b] - x
        x = max(0.0, x)
        y = max(0.0, y)

        # Create new node
        new_tree = f"({trees[a]},{trees[b]})"
        new_id_local = next_id
        next_id += 1

        # Compute new distances to remaining active nodes
        rest = [x for x in active if x != a and x != b]
        for k in rest:
            d_work[new_id_local, k] = 0.5 * (d_work[a, k] + d_work[b, k] - d_work[a, b])
            d_work[k, new_id_local] = d_work[new_id_local, k]

        # Update
        active = rest + [new_id_local]
        trees[new_id_local] = new_tree

    # Build final Newick
    if len(active) == 3:
        a, b, c = active
        result = f"({trees[a]},{trees[b]},{trees[c]})"
    elif len(active) == 2:
        a, b = active
        result = f"({trees[a]},{trees[b]})"
    else:
        result = "(" + ",".join(trees[x] for x in active) + ")"

    return result + ";"


# EVOLVE-BLOCK-END
