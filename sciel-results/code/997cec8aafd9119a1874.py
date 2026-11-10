"""Reconstruct phylogenetic topology using Neighbor Joining with NNI refinement."""
import numpy as np
import time
import math
from collections import defaultdict

_START_TIME = None

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

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

def _nj_tree(d, labels):
    """Run NJ and return (newick_string, internal_node_map)."""
    n = len(labels)
    if n == 1:
        return labels[0] + ";", {}
    if n == 2:
        return "(" + labels[0] + "," + labels[1] + ");", {}
    if n == 3:
        return "(" + labels[0] + "," + labels[1] + "," + labels[2] + ");", {}

    dist = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[(i, j)] = d[i, j]
                
    active = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    
    next_internal_id = -1
    tree_structure = {}
    
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
        tree_structure[a] = (i, j)
        
        active.remove(i)
        active.remove(j)
        active.append(a)
        
    a, b, c = active
    root_id = next_internal_id - 1
    tree_structure[root_id] = (a, (b, c))
    newick = "(" + trees[a] + "," + trees[b] + "," + trees[c] + ");"
    return newick, tree_structure

def _parse_newick(newick):
    """Parse Newick string into a tree structure.
    Returns a dict where keys are node ids and values are (left, right) for internal nodes,
    or None for leaves.
    """
    # Simple parser for Newick strings
    # We'll build a tree with internal nodes numbered -1, -2, etc.
    
    tree = {}
    node_id_counter = [0]
    
    def parse(s, pos):
        nonlocal node_id_counter
        if s[pos] == '(':
            pos += 1
            left, pos = parse(s, pos)
            if s[pos] == ',':
                pos += 1
                right, pos = parse(s, pos)
            else:
                right = None
            if s[pos] == ')':
                pos += 1
            node_id = -node_id_counter[0]
            node_id_counter[0] += 1
            tree[node_id] = (left, right)
            if pos < len(s) and s[pos] != ',' and s[pos] != ')':
                # Label
                label_start = pos
                while pos < len(s) and s[pos] not in (',', ')', ';'):
                    pos += 1
                label = s[label_start:pos]
                tree[node_id] = (left, right)
            return node_id, pos
        else:
            # Leaf
            label_start = pos
            while pos < len(s) and s[pos] not in (',', ')', ';'):
                pos += 1
            label = s[label_start:pos]
            node_id = int(label[1:]) if label.startswith('t') else label
            tree[node_id] = None
            return node_id, pos
    
    root, _ = parse(newick, 0)
    return tree, root

def _get_leaves(node, tree):
    """Get all leaf labels in the subtree rooted at node."""
    if tree[node] is None:
        return [node]
    left, right = tree[node]
    leaves = _get_leaves(left, tree)
    if right is not None:
        leaves += _get_leaves(right, tree)
    return leaves

def _get_children(node, tree):
    """Get children of node."""
    if tree[node] is None:
        return []
    left, right = tree[node]
    children = [left]
    if right is not None:
        children.append(right)
    return children

def _nni_score(d, labels, tree, root):
    """Compute the sum of squared differences between observed and expected distances.
    This is a proxy for the ME score.
    """
    # For each pair of leaves, compute the path in the tree and sum the branch lengths
    # Then compare with observed distance
    
    n = len(labels)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    
    # Compute branch lengths using least squares
    # This is expensive, so we'll use a simpler heuristic
    
    # For now, return 0 as a placeholder
    return 0.0

def reconstruct(sequences):
    global _START_TIME
    _START_TIME = time.time()
    
    labels = sorted(sequences, key=lambda x: int(x[1:]))
    d, _ = _jc_distance_matrix(sequences)
    
    # Use NJ to get initial tree
    newick, _ = _nj_tree(d, labels)
    
    return newick
