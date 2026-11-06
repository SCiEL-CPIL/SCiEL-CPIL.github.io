"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
'''Neighbour joining on Jukes-Cantor corrected distances (Saitou and Nei 1987).'''
import math
import time
import numpy as np

_START_TIME = None

def _get_elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _jc_distance(a, b, n):
    """Compute Jukes-Cantor corrected distance between two sequences."""
    diff = sum(1 for x, y in zip(a, b) if x != y)
    p = diff / n
    if p >= 0.7499:
        return 3.0
    if p <= 0.0:
        return 0.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)

def _compute_distance_matrix(sequences, labels):
    """Compute the full distance matrix."""
    n_labels = len(labels)
    n_sites = len(sequences[labels[0]])
    D = np.zeros((n_labels, n_labels))
    for i in range(n_labels):
        si = sequences[labels[i]]
        for j in range(i + 1, n_labels):
            sj = sequences[labels[j]]
            d = _jc_distance(si, sj, n_sites)
            D[i][j] = d
            D[j][i] = d
    return D

def _nj_newick(D, labels):
    """Perform Neighbor Joining and return Newick string."""
    n = len(labels)
    if n <= 2:
        return ",".join(labels) + ";"
    
    # Work with a copy
    D = D.copy()
    nodes = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    alive = set(range(n))
    
    next_id = n
    while len(alive) > 3:
        m = len(alive)
        # Compute row sums
        row_sums = np.zeros(m)
        alive_list = list(alive)
        for idx, a in enumerate(alive_list):
            for b in alive_list:
                if b != a:
                    row_sums[idx] += D[a][b]
        
        # Find best pair
        best_criterion = float('inf')
        best_i, best_j = -1, -1
        for i in range(m):
            for j in range(i + 1, m):
                a, b = alive_list[i], alive_list[j]
                criterion = (m - 2) * D[a][b] - row_sums[i] - row_sums[j]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_i, best_j = i, j
        
        a = alive_list[best_i]
        b = alive_list[best_j]
        
        # Create new node
        new_id = next_id
        next_id += 1
        
        # Compute distances from new node to all other alive nodes
        for k in range(m):
            x = alive_list[k]
            if x == a or x == b:
                continue
            D[new_id][x] = 0.5 * (D[a][x] + D[b][x] - D[a][b])
            D[x][new_id] = D[new_id][x]
        
        # Build the tree string
        trees[new_id] = f"({trees[a]},{trees[b]})"
        
        # Remove a and b, add new_id
        alive.discard(a)
        alive.discard(b)
        alive.add(new_id)
    
    # Combine remaining three nodes into a star
    remaining = list(alive)
    if len(remaining) == 3:
        return f"({trees[remaining[0]]},{trees[remaining[1]]},{trees[remaining[2]]});"
    elif len(remaining) == 2:
        return f"({trees[remaining[0]]},{trees[remaining[1]]});"
    else:
        return trees[remaining[0]] + ";"

def reconstruct(sequences):
    """Reconstruct unrooted tree topology from sequences using Neighbor Joining."""
    global _START_TIME
    _START_TIME = time.time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 2:
        return ",".join(labels) + ";"
    
    # Compute distance matrix
    D = _compute_distance_matrix(sequences, labels)
    
    # Check time budget
    if _get_elapsed() > 200:
        # Fallback: return a star tree
        return "(" + ",".join(labels) + ");"
    
    # Run NJ
    result = _nj_newick(D, labels)
    return result
# EVOLVE-BLOCK-END

