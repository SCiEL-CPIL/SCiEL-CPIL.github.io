<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
'''Neighbour joining on Jukes-Cantor corrected distances (Saitou and Nei 1987).'''
import math


def _distance(a, b):
    differences = sum(1 for x, y in zip(a, b) if x != y)
    p = differences / len(a)
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    nodes = list(labels)
    trees = {name: name for name in labels}
    matrix = {(a, b): _distance(sequences[a], sequences[b]) for a in labels for b in labels if a != b}
    while len(nodes) > 3:
        n = len(nodes)
        total = {a: sum(matrix[(a, b)] for b in nodes if b != a) for a in nodes}
        best = None
        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                criterion = (n - 2) * matrix[(a, b)] - total[a] - total[b]
                if best is None or criterion < best[0]:
                    best = (criterion, a, b)
        _, a, b = best
        joined = f"({trees[a]},{trees[b]})"
        new = f"n{len(nodes)}_{a}_{b}"
        rest = [x for x in nodes if x not in (a, b)]
        for x in rest:
            matrix[(new, x)] = matrix[(x, new)] = 0.5 * (matrix[(a, x)] + matrix[(b, x)] - matrix[(a, b)])
        nodes = rest + [new]
        trees[new] = joined
    return "(" + ",".join(trees[x] for x in nodes) + ");"
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
'''Vectorized Neighbor Joining with Jukes-Cantor correction and deadline guard.'''
import time
import numpy as np

_DEADLINE = 265.0
_START = time.time()


def _jc_distances(sequences, labels):
    """Compute pairwise JC-corrected distances using NumPy."""
    n = len(labels)
    seqs = [sequences[lab] for lab in labels]
    length = len(seqs[0])
    
    dna_arr = np.array([list(s) for s in seqs])  # (n, length)
    
    mismatches = np.sum(dna_arr[:, None, :] != dna_arr[None, :, :], axis=2)
    
    p = mismatches.astype(float) / length
    p_clipped = np.clip(p, 0, 0.7499)
    dist = -0.75 * np.log(1.0 - 4.0 * p_clipped / 3.0)
    
    np.fill_diagonal(dist, 0.0)
    
    return dist


def _nj_vectorized(D, labels):
    """
    Vectorized Neighbor Joining.
    D: (n, n) distance matrix
    labels: list of taxon labels
    Returns: Newick string
    """
    n = len(labels)
    if n <= 2:
        return "(" + ",".join(labels) + ");"
    
    # D_active: current distance matrix
    # active_labels: list of current node labels
    # active_trees: list of Newick strings for each active node (index-aligned with active_labels)
    
    D_active = D.copy()
    active_labels = list(labels)
    active_trees = list(labels)  # each is its own label initially
    
    next_id = n
    
    while len(active_labels) > 3:
        m = len(active_labels)
        
        if time.time() > _DEADLINE:
            return "(" + ",".join(active_labels) + ");"
        
        # Row sums
        r = np.sum(D_active, axis=1)
        
        # Q matrix
        Q = (m - 2) * D_active - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        
        # Find min in upper triangle
        i_idx, j_idx = np.triu_indices(m, k=1)
        Q_upper = Q[i_idx, j_idx]
        min_pos = np.argmin(Q_upper)
        i, j = i_idx[min_pos], j_idx[min_pos]
        
        # New node
        new_label = f"n{next_id}"
        next_id += 1
        new_tree = f"({active_trees[i]},{active_trees[j]})"
        
        # Indices of other active nodes
        others = [k for k in range(m) if k != i and k != j]
        
        # New distances from new node to others
        if others:
            d_ij = D_active[i, j]
            d_new_others = 0.5 * (D_active[i, others] + D_active[j, others] - d_ij)
        else:
            d_new_others = np.array([])
        
        # Build new active set
        new_labels = [active_labels[k] for k in others] + [new_label]
        new_trees = [active_trees[k] for k in others] + [new_tree]
        
        # Build new distance matrix
        new_m = len(others) + 1
        D_new = np.zeros((new_m, new_m))
        
        if others:
            D_new[:len(others), :len(others)] = D_active[np.ix_(others, others)]
            D_new[:len(others), len(others)] = d_new_others
            D_new[len(others), :len(others)] = d_new_others
        
        D_active = D_new
        active_labels = new_labels
        active_trees = new_trees
    
    return "(" + ",".join(active_trees) + ");"


def reconstruct(sequences):
    global _START, _DEADLINE
    
    _START = time.time()
    _DEADLINE = _START + 265.0
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 2:
        return "(" + ",".join(labels) + ");"
    
    D = _jc_distances(sequences, labels)
    newick = _nj_vectorized(D, labels)
    
    return newick
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
