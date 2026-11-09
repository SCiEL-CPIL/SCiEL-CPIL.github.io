"""Phylogenetic topology recovery using NJ and Parsimony refinement."""
import time
import math
import numpy as np

_start_time = time.time()
_BUDGET = 170.0

def _time_remaining():
    return _BUDGET - (time.time() - _start_time)

def _jc_distance_matrix(seqs_array):
    n_taxa, n_sites = seqs_array.shape
    diff = np.zeros((n_taxa, n_taxa), dtype=np.int32)
    for k in range(n_sites):
        col = seqs_array[:, k]
        diff += (col[:, None] != col[None, :]).astype(np.int32)
    p = diff / n_sites
    with np.errstate(divide='ignore', invalid='ignore'):
        d = -0.75 * np.log(1.0 - 4.0/3.0 * p)
    d = np.where(np.isfinite(d), d, 3.0)
    np.fill_diagonal(d, 0.0)
    return d

def _neighbor_joining(dist):
    n = dist.shape[0]
    if n < 3:
        return {}
        
    max_nodes = 2 * n
    D = np.zeros((max_nodes, max_nodes), dtype=np.float64)
    D[:n, :n] = dist
    
    active = list(range(n))
    tree = {}
    next_id = n
    
    while len(active) > 2:
        m = len(active)
        idx = np.array(active)
        D_sub = D[np.ix_(idx, idx)]
        
        r = D_sub.sum(axis=1)
        Q = (m - 2) * D_sub - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        
        min_val = np.min(Q)
        i, j = np.unravel_index(np.argmin(Q), Q.shape)
        
        node_i = active[i]
        node_j = active[j]
        
        new_node = next_id
        next_id += 1
        
        others = [x for x in active if x not in (node_i, node_j)]
        if others:
            others_idx = np.array(others)
            D[new_node, others_idx] = 0.5 * (D[node_i, others_idx] + D[node_j, others_idx] - D[node_i, node_j])
            D[others_idx, new_node] = D[new_node, others_idx]
            
        active.remove(node_i)
        active.remove(node_j)
        active.append(new_node)
        
        tree[new_node] = (node_i, node_j)
        
    if len(active) == 3:
        a, b, c = active
        node1 = next_id
        next_id += 1
        node2 = next_id
        next_id += 1
        
        tree[node1] = (a, b)
        tree[node2] = (node1, c)
        
    return tree

def _get_root(tree, n_taxa):
    children_set = set()
    for children in tree.values():
        children_set.add(children[0])
        children_set.add(children[1])
        
    all_nodes = set(tree.keys()) | children_set
    roots = all_nodes - children_set
    return list(roots)[0]

def _parsimony_score(tree, seqs_array, root_node):
    n_taxa, n_sites = seqs_array.shape
    leaves = set(range(n_taxa))
    
    post_order = []
    visited = set()
    
    def dfs(node):
        if node in leaves:
            post_order.append(node)
            return
        if node in visited:
            return
        visited.add(node)
        left, right = tree[node]
        dfs(left)
        dfs(right)
        post_order.append(node)
        
    dfs(root_node)
    
    states = {}
    for leaf in leaves:
        states[leaf] = 1 << seqs_array[leaf]
        
    total_score = 0
    for node in post_order:
        if node in leaves:
            continue
        left, right = tree[node]
        s_left = states[left]
        s_right = states[right]
        
        intersection = s_left & s_right
        union = s_left | s_right
        
        change_mask = (intersection == 0)
        total_score += np.sum(change_mask)
        
        states[node] = np.where(change_mask, union, intersection)
        
    return total_score

def _nni_refine(tree, seqs_array, n_taxa):
    root = _get_root(tree, n_taxa)
    current_score = _parsimony_score(tree, seqs_array, root)
    
    leaves = set(range(n_taxa))
    
    while _time_remaining() > 5.0:
        improved = False
        
        # Get internal edges
        internal_edges = []
        for parent, (left, right) in tree.items():
            if left not in leaves:
                internal_edges.append((parent, left))
            if right not in leaves:
                internal_edges.append((parent, right))
                
        for u, v in internal_edges:
            left, right = tree[u]
            if left == v:
                w = right
            else:
                w = left
                
            x, y = tree[v]
            
            # Alternative 1: ((x, w), y)
            # u -> (new1, y), new1 -> (x, w)
            
            # Check if w is a leaf?
            # If w is a leaf, NNI is still valid.
            
            # Save state
            orig_u = tree[u]
            orig_v = tree[v]
            
            new_node = max(tree.keys()) + 1 if tree else n_taxa + 100
            while new_node in tree or new_node < n_taxa:
                new_node += 1
                
            # Apply change
            tree[new_node] = (x, w)
            tree[u] = (new_node, y)
            del tree[v]
            
            # New root?
            # If root was v, it's gone.
            # If root was u, it's still u.
            # If root was something else, it's still that.
            # We need to find the new root.
            new_root = _get_root(tree, n_taxa)
            
            new_score = _parsimony_score(tree, seqs_array, new_root)
            
            if new_score < current_score:
                current_score = new_score
                root = new_root
                improved = True
            else:
                # Restore
                tree[u] = orig_u
                tree[v] = orig_v
                del tree[new_node]
                
        if not improved:
            break
            
    return tree

def reconstruct(sequences):
    if _time_remaining() < 1.0:
        labels = sorted(sequences, key=lambda name: int(name[1:]))
        return "(" + ",".join(labels) + ");"
        
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    seq_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seqs_array = np.array([[seq_map[base] for base in seq] for seq in (sequences[label] for label in labels)], dtype=np.uint8)
    
    dist = _jc_distance_matrix(seqs_array)
    tree = _neighbor_joining(dist)
    
    if n > 3:
        tree = _nni_refine(tree, seqs_array, n)
        
    root = _get_root(tree, n)
    
    def to_newick(node):
        if node < n:
            return labels[node]
        left, right = tree[node]
        return f"({to_newick(left)},{to_newick(right)})"
        
    return to_newick(root) + ";"
