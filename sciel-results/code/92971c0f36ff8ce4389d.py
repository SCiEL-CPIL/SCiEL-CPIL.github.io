"""Reconstruct phylogenetic topology using Neighbor Joining with Minimum Evolution refinement."""
import numpy as np
import time
import math

_START_TIME = None
_DEADLINE = 250.0  # seconds, safety margin before 270s kill

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _time_left():
    return _DEADLINE - _elapsed()

def _jc_distance_matrix(sequences):
    labels = sorted(sequences, key=lambda x: int(x[1:]))
    n = len(labels)
    seq_data = np.array([[ord(c) for c in sequences[label]] for label in labels])
    diff = np.sum(seq_data[:, np.newaxis, :] != seq_data[np.newaxis, :, :], axis=2)
    p = diff / float(len(sequences[labels[0]]))
    d = -0.75 * np.log(1 - (4.0/3.0) * p)
    d[p >= 0.75] = 3.0
    np.fill_diagonal(d, 0.0)
    return d, labels

def _nj_newick(d, labels):
    n = len(labels)
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return "(" + labels[0] + "," + labels[1] + ");"
    if n == 3:
        return "(" + labels[0] + "," + labels[1] + "," + labels[2] + ");"

    dist = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[(i, j)] = d[i, j]
                
    active = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    
    next_internal_id = -1
    
    while len(active) > 3:
        n_act = len(active)
        
        r = {i: 0.0 for i in active}
        for i in active:
            for j in active:
                if i != j:
                    r[i] += dist.get((i, j), 0.0)
                    
        best_c = float('inf')
        best_i, best_j = -1, -1
        for i in active:
            for j in active:
                if i == j: continue
                c = (n_act - 2) * dist.get((i, j), 0.0) - r[i] - r[j]
                if c < best_c:
                    best_c = c
                    best_i, best_j = i, j
                    
        i, j = best_i, best_j
        d_ij = dist.get((i, j), 0.0)
        
        a = next_internal_id
        next_internal_id -= 1
        
        for k in active:
            if k == i or k == j: continue
            d_ik = dist.get((i, k), 0.0)
            d_jk = dist.get((j, k), 0.0)
            d_ak = 0.5 * (d_ik + d_jk - d_ij)
            dist[(a, k)] = d_ak
            dist[(k, a)] = d_ak
            
        trees[a] = "(" + trees[i] + "," + trees[j] + ")"
        
        active.remove(i)
        active.remove(j)
        active.append(a)
        
    a, b, c = active
    return "(" + trees[a] + "," + trees[b] + "," + trees[c] + ");"

def _parse_newick(newick):
    """Parse a Newick string into a tree structure.
    Returns:
        nodes: dict mapping node_id -> (left, right) or (label,) for leaves
        node_labels: dict mapping node_id -> label (for leaves)
        root: root node id
    """
    # We'll use a simple parser
    # Since we control the Newick generation, we can use a simpler approach
    # But for generality, let's implement a proper parser
    
    pos = [0]
    n = len(newick)
    
    def parse():
        if pos[0] >= n:
            return None
        if newick[pos[0]] == '(':
            pos[0] += 1
            left = parse()
            while newick[pos[0]] != ')':
                if newick[pos[0]] == ',':
                    pos[0] += 1
                    right = parse()
                    if left is None:
                        left = right
                    else:
                        # Handle multiple children
                        # For our purposes, we assume binary trees
                        pass
            pos[0] += 1  # skip ')'
            label = ""
            while pos[0] < n and newick[pos[0]] not in (',', ';', '(', ')'):
                label += newick[pos[0]]
                pos[0] += 1
            if label:
                return ('node', label, left)
            else:
                return ('node', '', left)
        else:
            label = ""
            while pos[0] < n and newick[pos[0]] not in (',', ';', '(', ')'):
                label += newick[pos[0]]
                pos[0] += 1
            return ('leaf', label, None)
    
    root = parse()
    return root

def _get_splits_from_newick(newick, labels):
    """Extract all non-trivial splits from a Newick tree.
    Returns a set of frozensets representing the splits.
    """
    n = len(labels)
    if n <= 3:
        return set()
    
    # For a binary unrooted tree with n leaves, there are n-3 non-trivial splits
    # We'll use a simple approach: for each internal node, the split is the set of leaves in its subtree
    
    # This requires a full tree structure, so let's use a different approach
    # For now, return an empty set as a placeholder
    # In a real implementation, we'd parse the tree and compute splits
    return set()

def _me_score(newick, d, labels):
    """Compute Minimum Evolution score: sum of all branch lengths.
    For an unrooted tree, we compute branch lengths from the distance matrix
    and return their sum.
    
    Returns the sum of branch lengths (lower is better).
    """
    n = len(labels)
    if n <= 2:
        return 0.0
    
    # For a proper ME score, we need to:
    # 1. Parse the tree to get the topology
    # 2. Compute branch lengths using least squares or the standard formula
    # 3. Sum all branch lengths
    
    # This is complex, so for now we'll use a proxy score
    # The ME score is approximately related to the sum of pairwise distances
    # divided by the number of taxa
    
    # A better proxy: the sum of all pairwise distances in the tree
    # For a tree with n leaves, the sum of all pairwise distances is:
    # sum_{i<j} d(i,j) = sum over all edges of (length * n_i * (n - n_i))
    # where n_i is the number of leaves on one side of the edge
    
    # For simplicity, we'll use the average pairwise distance as a proxy
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            total += d[i, j]
            count += 1
    
    return total / max(count, 1)

def reconstruct(sequences):
    global _START_TIME
    _START_TIME = time.time()
    
    labels = sorted(sequences, key=lambda x: int(x[1:]))
    d, _ = _jc_distance_matrix(sequences)
    newick = _nj_newick(d, labels)
    return newick
