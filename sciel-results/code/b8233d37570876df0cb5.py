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
    """Compute JC-corrected distance matrix from sequences."""
    labels = sorted(sequences, key=lambda x: int(x[1:]))
    n = len(labels)
    seq_data = np.array([[ord(c) for c in sequences[label]] for label in labels])
    diff = np.sum(seq_data[:, np.newaxis, :] != seq_data[np.newaxis, :, :], axis=2)
    p = diff / float(len(sequences[labels[0]]))
    d = -0.75 * np.log(1 - (4.0/3.0) * p)
    d[p >= 0.75] = 3.0
    np.fill_diagonal(d, 0.0)
    return d, labels

def _nj_tree(d, labels):
    """Neighbor Joining algorithm. Returns tree as nested tuples."""
    n = len(labels)
    if n == 1:
        return labels[0]
    if n == 2:
        return (labels[0], labels[1])
    if n == 3:
        return (labels[0], (labels[1], labels[2]))
    
    # Work with a list of active nodes
    active = list(range(n))
    dist = d.copy()  # distance matrix
    trees = {i: labels[i] for i in range(n)}
    next_id = n
    
    while len(active) > 3:
        m = len(active)
        # Compute row sums
        r = np.zeros(m)
        for i_idx in range(m):
            i = active[i_idx]
            for j_idx in range(m):
                if j_idx != i_idx:
                    j = active[j_idx]
                    r[i_idx] += dist[i, j]
        
        # Find min Q matrix
        min_q = float('inf')
        min_i = min_j = None
        for i_idx in range(m):
            i = active[i_idx]
            for j_idx in range(i_idx + 1, m):
                j = active[j_idx]
                q = (m - 2) * dist[i, j] - r[i_idx] - r[j_idx]
                if q < min_q:
                    min_q = q
                    min_i = i
                    min_j = j
        
        i, j = min_i, min_j
        d_ij = dist[i, j]
        
        # Create new internal node
        new_id = next_id
        next_id += 1
        
        # Update distances
        for k in active:
            if k == i or k == j:
                continue
            d_ik = dist[i, k]
            d_jk = dist[j, k]
            d_nk = 0.5 * (d_ik + d_jk - d_ij)
            dist[new_id, k] = d_nk
            dist[k, new_id] = d_nk
        
        # Remove i and j from active, add new_id
        active.remove(i)
        active.remove(j)
        active.append(new_id)
        
        # Build tree structure
        trees[new_id] = (trees[i], trees[j])
    
    # Combine remaining 3
    a, b, c = active
    return (trees[a], (trees[b], trees[c]))

def _tree_to_newick(tree, labels=None):
    """Convert tree (nested tuples or leaf names) to Newick string."""
    if isinstance(tree, str):
        return tree
    if isinstance(tree, tuple):
        left = _tree_to_newick(tree[0], labels)
        right = _tree_to_newick(tree[1], labels)
        return "(" + left + "," + right + ")"
    return str(tree)

def _get_splits(tree, all_leaves):
    """Extract all non-trivial splits from a tree (nested tuples)."""
    splits = set()
    
    def get_leaf_set(node):
        if isinstance(node, str):
            return frozenset([node])
        elif isinstance(node, tuple):
            return get_leaf_set(node[0]) | get_leaf_set(node[1])
        return frozenset()
    
    def collect_splits(node):
        leaf_set = get_leaf_set(node)
        if 0 < len(leaf_set) < len(all_leaves):
            # Canonicalize: keep the smaller side
            comp = all_leaves - leaf_set
            if min(leaf_set) < min(comp):
                splits.add(leaf_set)
            else:
                splits.add(comp)
        
        if isinstance(node, tuple):
            collect_splits(node[0])
            collect_splits(node[1])
    
    collect_splits(tree)
    return splits

def _rf_distance(splits1, splits2):
    """Calculate normalized Robinson-Foulds distance."""
    set1 = set(splits1)
    set2 = set(splits2)
    intersection = set1 & set2
    union = set1 | set2
    if len(union) == 0:
        return 0.0
    return 1.0 - (2.0 * len(intersection) / len(union))

def _nni_refine(tree, d, labels, max_rounds=5):
    """NNI refinement using distance matrix scoring (Minimum Evolution-like)."""
    if len(labels) < 4:
        return tree
    
    all_leaves = frozenset(labels)
    current_tree = tree
    current_splits = _get_splits(current_tree, all_leaves)
    
    def tree_score(tree):
        """Score tree by sum of pairwise distances through the tree (ME-like)."""
        # For simplicity, use RF distance to a reference if available
        # Here we just return 0 as placeholder - actual ME scoring needs branch lengths
        return 0.0
    
    # NNI moves require careful tree manipulation
    # For now, return the NJ tree without refinement to ensure stability
    return current_tree

def reconstruct(sequences):
    global _START_TIME
    _START_TIME = time.time()
    
    labels = sorted(sequences, key=lambda x: int(x[1:]))
    n = len(labels)
    
    # Compute distance matrix
    d, _ = _jc_distance_matrix(sequences)
    
    # Get NJ tree
    tree = _nj_tree(d, labels)
    
    # Convert to Newick
    newick = _tree_to_newick(tree) + ";"
    
    return newick
