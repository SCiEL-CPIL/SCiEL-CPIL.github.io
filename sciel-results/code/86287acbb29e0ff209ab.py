"""Phylogenetic topology recovery using Neighbor Joining with NumPy (fixed JC distance)."""
import time
import numpy as np

# Deadline guard: 270s wall-clock limit, use 250s as safety margin
_DEADLINE = time.time() + 250.0


def _time_exceeded():
    return time.time() > _DEADLINE


def _encode_sequences(seqs):
    """Map each DNA string to int8 array (A=0,C=1,G=2,T=3)."""
    n = len(seqs)
    L = len(seqs[0]) if n else 0
    if n == 0 or L == 0:
        return np.zeros((n, L), dtype=np.int8)
    out = np.zeros((n, L), dtype=np.int8)
    # Vectorized per-base mapping using np.where (no huge literal lookup)
    for i, s in enumerate(seqs):
        a = np.frombuffer(s.encode('ascii'), dtype=np.uint8)
        v = np.where(a == 65, 0,
            np.where(a == 67, 1,
            np.where(a == 71, 2,
            np.where(a == 84, 3,
            np.where(a == 97, 0,
            np.where(a == 99, 1,
            np.where(a == 103, 2, 3)))))))
        out[i] = v.astype(np.int8)
    return out


def _jc_distance_matrix(sequences, labels):
    """Pairwise Jukes-Cantor corrected distances (vectorized)."""
    n = len(labels)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    seqs = [sequences[label] for label in labels]
    L = len(seqs[0]) if seqs else 0
    if L == 0:
        return np.zeros((n, n), dtype=np.float64)

    X = _encode_sequences(seqs)  # (n, L) int8
    # Hamming distances via per-base XOR popcount (4 passes, less memory than broadcast !=)
    H = np.zeros((n, n), dtype=np.int32)
    for b in range(4):
        col = (X == b)  # (n, L) bool
        # XOR matrix: rows differ at base b
        xor = col[:, None] ^ col[None, :]
        H += xor.sum(axis=1).astype(np.int32)

    D = np.zeros((n, n), dtype=np.float64)
    p = H / L
    # JC: d = -0.75 * ln(1 - 4p/3), clip p
    pc = np.minimum(p, 0.7499)
    d = -0.75 * np.log(1.0 - 4.0 * pc / 3.0)
    d[p >= 0.7499] = 3.0
    np.fill_diagonal(d, 0.0)
    return d


def _neighbor_joining(D, labels):
    """Neighbor Joining; returns Newick (topology)."""
    n = len(labels)
    if n <= 1:
        return labels[0] + ";"
    if n == 2:
        return "(" + labels[0] + "," + labels[1] + ");"

    dist = D.copy()
    tree_parts = list(labels)
    num_nodes = n

    while num_nodes > 3:
        if _time_exceeded():
            break
        m = num_nodes
        r = dist.sum(axis=1)
        criteria = (m - 2) * dist - r[:, None] - r[None, :]
        np.fill_diagonal(criteria, np.inf)
        iu = np.triu_indices(m, k=1)
        flat = criteria[iu]
        best = int(np.argmin(flat))
        i, j = int(iu[0][best]), int(iu[1][best])

        new_part = "(" + tree_parts[i] + "," + tree_parts[j] + ")"
        keep = [k for k in range(m) if k != i and k != j]
        new_distances = 0.5 * (dist[i, keep] + dist[j, keep] - dist[i, j])

        new_dist = np.empty((m - 1, m - 1), dtype=np.float64)
        new_dist[:-1, :-1] = dist[np.ix_(keep, keep)]
        new_dist[:-1, -1] = new_distances
        new_dist[-1, :-1] = new_distances
        new_dist[-1, -1] = 0.0

        tree_parts = [tree_parts[k] for k in keep] + [new_part]
        dist = new_dist
        num_nodes = m - 1

    result = "(" + ",".join(tree_parts) + ")"
    return result + ";"


def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    if n <= 1:
        return labels[0] + ";"
    if n == 2:
        return "(" + labels[0] + "," + labels[1] + ");"
    D = _jc_distance_matrix(sequences, labels)
    return _neighbor_joining(D, labels)
