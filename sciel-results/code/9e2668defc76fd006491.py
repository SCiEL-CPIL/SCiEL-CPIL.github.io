"""Phylogenetic topology recovery via Neighbor Joining under JC69."""
import math
import random
import itertools
from collections import defaultdict

import numpy as np

def _neighbor_joining(sequences, leaf_order, n_taxa):
    """
    Perform Neighbor Joining to get an initial tree topology.
    Returns a Tree object.
    """
    n_sites = len(sequences[leaf_order[0]])
    
    # Compute pairwise Hamming distances
    # Convert sequences to numeric matrix
    seq_matrix = np.zeros((n_sites, n_taxa), dtype=np.int8)
    for idx, label in enumerate(leaf_order):
        seq = sequences[label]
        s = np.array(list(seq), dtype='S1').view(np.uint8)
        seq_matrix[:, idx] = np.where(s == 65, 0, np.where(s == 67, 1, np.where(s == 71, 2, 3)))
        
    # Hamming distance
    diff = (seq_matrix[:, :, None] != seq_matrix[:, None, :])
    dist_matrix = np.sum(diff, axis=0)
    
    # Neighbor Joining
    tips = list(range(n_taxa))
    dist = dist_matrix.copy()
    
    # Tree structure
    # We will build the tree by keeping track of the nodes created.
    # Node IDs: 0..n_taxa-1 are leaves.
    # n_taxa..2*n_taxa-2 are internal nodes.
    
    n_nodes = 2 * n_taxa - 1
    left = np.full(n_nodes, -1, dtype=np.int32)
    right = np.full(n_nodes, -1, dtype=np.int32)
    leaf_idx = np.full(n_nodes, -1, dtype=np.int32)
    
    for i in range(n_taxa):
        leaf_idx[i] = i
        
    next_internal_id = n_taxa
    
    while len(tips) > 2:
        n_tips = len(tips)
        # Compute Q matrix
        r = np.sum(dist, axis=1)
        Q = (n_tips - 2) * dist - r[:, None] - r[None, :]
        
        # Find min Q(i, j) with i < j
        min_val = np.inf
        min_i, min_j = 0, 1
        for i in range(n_tips):
            for j in range(i + 1, n_tips):
                if Q[i, j] < min_val:
                    min_val = Q[i, j]
                    min_i, min_j = i, j
        
        i, j = min_i, min_j
        node_i = tips[i]
        node_j = tips[j]
        
        # Create new internal node
        new_node = next_internal_id
        next_internal_id += 1
        
        # Connect new_node to node_i and node_j
        left[new_node] = node_i
        right[new_node] = node_j
        leaf_idx[new_node] = -1
        
        # Update distance matrix
        # Remove i and j, add new_node
        
        # Calculate new distances from new_node to other tips
        # d(new_node, k) = 0.5 * (d(i, k) + d(j, k) - d(i, j))
        
        # We need to update the dist matrix for the remaining tips
        new_tips = [t for idx, t in enumerate(tips) if idx != i and idx != j]
        new_tips.append(new_node)
        
        new_dist = np.zeros((len(new_tips), len(new_tips)))
        
        # Copy old distances between remaining nodes
        # And compute new distances to new_node
        
        for idx1, t1 in enumerate(new_tips):
            if t1 == new_node:
                continue
            old_idx_t1 = tips.index(t1)
            for idx2, t2 in enumerate(new_tips):
                if t1 == t2:
                    new_dist[idx1, idx2] = 0
                elif t2 == new_node:
                    d_ik = dist[old_idx_t1, i]
                    d_jk = dist[old_idx_t1, j]
                    d_ij = dist[i, j]
                    new_dist[idx1, idx2] = 0.5 * (d_ik + d_jk - d_ij)
                else:
                    old_idx_t2 = tips.index(t2)
                    new_dist[idx1, idx2] = dist[old_idx_t1, old_idx_t2]
        
        # Symmetrize
        new_dist = (new_dist + new_dist.T) / 2.0
        np.fill_diagonal(new_dist, 0)
        
        tips = new_tips
        dist = new_dist
        
    # Two nodes remain
    node_a = tips[0]
    node_b = tips[1]
    
    # Create root
    root_node = next_internal_id
    left[root_node] = node_a
    right[root_node] = node_b
    leaf_idx[root_node] = -1
    
    # Now we have a tree with root at root_node.
    # We need to relabel so that root is 0.
    
    new_left = np.full(n_nodes, -1, dtype=np.int32)
    new_right = np.full(n_nodes, -1, dtype=np.int32)
    new_leaf = np.full(n_nodes, -1, dtype=np.int32)
    
    old_to_new = {}
    new_id_counter = [0]
    
    def relabel(old_id):
        new_id = new_id_counter[0]
        new_id_counter[0] += 1
        old_to_new[old_id] = new_id
        
        if leaf_idx[old_id] >= 0:
            new_leaf[new_id] = leaf_idx[old_id]
            new_left[new_id] = -1
            new_right[new_id] = -1
        else:
            l_old = left[old_id]
            r_old = right[old_id]
            l_new = relabel(l_old)
            r_new = relabel(r_old)
            new_left[new_id] = l_new
            new_right[new_id] = r_new
            new_leaf[new_id] = -1
            
        return new_id
    
    relabel(root_node)
    
    # Create a simple object to hold the tree
    class TreeObj:
        def __init__(self, left, right, leaf):
            self.left = left
            self.right = right
            self.leaf = leaf
            self.n_taxa = n_taxa
            
        def to_newick(self, leaf_order):
            def get_node_str(node):
                if self.leaf[node] >= 0:
                    return leaf_order[self.leaf[node]]
                else:
                    l_str = get_node_str(self.left[node])
                    r_str = get_node_str(self.right[node])
                    return f"({l_str},{r_str})"
            return get_node_str(0) + ";"
            
    return TreeObj(new_left, new_right, new_leaf)

def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n_taxa = len(labels)
    n_sites = len(sequences[labels[0]])
    
    if n_taxa <= 1:
        return labels[0] + ";" if labels else ";";
    if n_taxa == 2:
        return f"({labels[0]},{labels[1]});"
    if n_taxa == 3:
        return f"({labels[0]},{labels[1]},{labels[2]});"
    
    tree = _neighbor_joining(sequences, labels, n_taxa)
    return tree.to_newick(labels)
