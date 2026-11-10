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
'''Neighbour joining on Jukes-Cantor corrected distances (Saitou and Nei 1987).'''
import math
import time
import numpy as np


def _jc_distances(sequences, labels):
    """Compute pairwise JC-corrected distances using vectorized operations."""
    n = len(labels)
    # Convert sequences to integer arrays for fast comparison
    seq_array = np.array([list(s) for s in sequences], dtype='U1')
    # Compute pairwise differences
    # For each pair (i, j), count mismatches
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            diffs = np.sum(seq_array[i] != seq_array[j])
            p = diffs / len(sequences[labels[0]])
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
    return dist_matrix


def _nj_tree(dist_matrix, labels):
    """Neighbor Joining algorithm on a distance matrix."""
    n = len(labels)
    if n <= 1:
        return labels[0] if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},({labels[1]},{labels[2]}))"
    
    # Work with a copy
    D = dist_matrix.copy()
    names = list(labels)
    tree = {name: name for name in names}
    
    while len(names) > 3:
        m = len(names)
        # Compute row sums
        r = D.sum(axis=1)
        # Compute Q matrix: Q[i,j] = (m-2)*D[i,j] - r[i] - r[j]
        Q = (m - 2) * D - r[:, None] - r[None, :]
        # Find minimum Q value (off-diagonal)
        np.fill_diagonal(Q, np.inf)
        min_idx = np.unravel_index(np.argmin(Q), Q.shape)
        i, j = min_idx[0], min_idx[1]
        
        # Join i and j
        new_name = f"n{m}_{names[i]}_{names[j]}"
        tree[new_name] = f"({tree[names[i]]},{tree[names[j]]})"
        
        # Compute new distances
        new_dist = np.zeros(m - 1)
        for k, idx in enumerate([x for x in range(m) if x not in (i, j)]):
            new_dist[k] = 0.5 * (D[i, idx] + D[j, idx] - D[i, j])
        
        # Build new distance matrix
        new_names = [names[x] for x in range(m) if x not in (i, j)] + [new_name]
        new_D = np.zeros((m - 1, m - 1))
        for a in range(m - 1):
            for b in range(m - 1):
                if a == b:
                    new_D[a, b] = 0
                elif a == m - 2:  # new node
                    new_D[a, b] = new_dist[b]
                elif b == m - 2:
                    new_D[a, b] = new_dist[a]
                else:
                    old_a = a if a < i and a < j else (a - 1 if a >= i and a >= j else a)
                    # Need to map old indices to new
                    # Let's do this more carefully
                    pass
        
        # Simpler approach: rebuild D
        old_names = names
        names = [old_names[x] for x in range(m) if x not in (i, j)] + [new_name]
        new_D = np.zeros((m - 1, m - 1))
        for a_idx in range(m - 1):
            for b_idx in range(m - 1):
                if a_idx == b_idx:
                    new_D[a_idx, b_idx] = 0
                elif a_idx == m - 2:
                    # new node to old node
                    old_idx = [x for x in range(m) if x not in (i, j)][b_idx]
                    new_D[a_idx, b_idx] = 0.5 * (D[i, old_idx] + D[j, old_idx] - D[i, j])
                elif b_idx == m - 2:
                    old_idx = [x for x in range(m) if x not in (i, j)][a_idx]
                    new_D[a_idx, b_idx] = 0.5 * (D[i, old_idx] + D[j, old_idx] - D[i, j])
                else:
                    old_a = [x for x in range(m) if x not in (i, j)][a_idx]
                    old_b = [x for x in range(m) if x not in (i, j)][b_idx]
                    new_D[a_idx, b_idx] = D[old_a, old_b]
        
        D = new_D
    
    # Final three nodes
    return f"({tree[names[0]]},({tree[names[1]]},{tree[names[2]]}))"


def reconstruct(sequences):
    start_time = time.time()
    deadline = start_time + 250  # 250s safety margin
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    # Compute distance matrix
    dist_matrix = _jc_distances(sequences, labels)
    
    # Run NJ
    result = _nj_tree(dist_matrix, labels)
    
    return result
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
