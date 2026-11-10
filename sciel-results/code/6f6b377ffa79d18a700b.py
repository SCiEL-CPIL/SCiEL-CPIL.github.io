import numpy as np
import math
import time

def reconstruct(sequences):
    start_time = time.time()
    deadline = start_time + 165.0

    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n_taxa = len(labels)

    if n_taxa <= 2:
        if n_taxa == 1:
            return f"({labels[0]});"
        else:
            return f"({labels[0]},{labels[1]});"

    # Map sequences to integer arrays
    seq_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    L = len(next(iter(sequences.values())))
    
    seq_array = np.empty((n_taxa, L), dtype=np.int8)
    for i, label in enumerate(labels):
        s = sequences[label]
        seq_array[i] = [seq_map[c] for c in s]

    # Compute pairwise Hamming distances
    diff = (seq_array[:, None, :] != seq_array[None, :, :])
    hamming = diff.sum(axis=2)
    
    p = hamming / L
    
    d_matrix = np.zeros((n_taxa, n_taxa), dtype=np.float64)
    valid_mask = (p > 0) & (p < 0.75)
    
    if np.any(valid_mask):
        d_valid = -0.75 * np.log(1 - (4.0 / 3.0) * p[valid_mask])
        d_matrix[valid_mask] = d_valid
        large_val = np.max(d_valid) * 1.5
    else:
        large_val = 1.0
        
    invalid_mask = p >= 0.75
    d_matrix[invalid_mask] = large_val
    np.fill_diagonal(d_matrix, 0.0)

    # Neighbor Joining Algorithm
    # active_nodes: list of currently active node indices (original taxa + new internal nodes)
    # tree_edges: dictionary mapping node_id -> list of (neighbor_id, length)
    # node_labels: dictionary mapping node_id -> label string (for leaves) or None (for internal)
    
    active_nodes = list(range(n_taxa))
    tree_edges = {i: [] for i in active_nodes}
    node_labels = {i: labels[i] for i in active_nodes}
    
    next_node_id = n_taxa
    d = d_matrix.copy()
    
    while len(active_nodes) > 2:
        if time.time() > deadline - 5.0:
            break
            
        n = len(active_nodes)
        if n == 0:
            break
            
        # Calculate Q matrix for active nodes
        # Q(i,j) = (n-2)*d(i,j) - r(i) - r(j)
        # where r(i) = sum_k d(i,k)
        
        # Extract submatrix of d for active nodes
        # This is O(n^2) to build, but n is small (max 32)
        # We can optimize by just using the indices directly
        
        r = np.zeros(n)
        for i in range(n):
            r[i] = sum(d[active_nodes[i], active_nodes[k]] for k in range(n) if k != i)
        
        max_q = -float('inf')
        best_i, best_j = -1, -1
        
        for i in range(n):
            for j in range(i+1, n):
                q = (n - 2) * d[active_nodes[i], active_nodes[j]] - r[i] - r[j]
                if q > max_q:
                    max_q = q
                    best_i = i
                    best_j = j
        
        if best_i == -1:
            break
            
        node_i = active_nodes[best_i]
        node_j = active_nodes[best_j]
        
        # Calculate branch lengths
        # delta = r[i] - r[j] / (n-2)
        delta = (r[best_i] - r[best_j]) / (n - 2)
        d_ij = d[node_i, node_j]
        
        len_i = 0.5 * d_ij + 0.5 * delta
        len_j = 0.5 * d_ij - 0.5 * delta
        
        # Clamp lengths to be non-negative
        if len_i < 0: len_i = 0.0
        if len_j < 0: len_j = 0.0
        
        # Create new internal node
        new_node = next_node_id
        next_node_id += 1
        
        # Add edges
        tree_edges[node_i].append((new_node, len_i))
        tree_edges[node_j].append((new_node, len_j))
        tree_edges[new_node] = []
        node_labels[new_node] = None
        
        # Update active nodes
        active_nodes.remove(node_i)
        active_nodes.remove(node_j)
        active_nodes.append(new_node)
        
        # Update distance matrix for the new node
        # d(new, k) = 0.5 * (d(i, k) + d(j, k) - d(i, j))
        for k in active_nodes:
            if k == new_node:
                continue
            d[new_node, k] = 0.5 * (d[node_i, k] + d[node_j, k] - d_ij)
            d[k, new_node] = d[new_node, k]
            
        # Ensure symmetry and zero diagonal (though we don't use diagonal)
        # d[new_node, new_node] = 0.0

    # If we broke out early or ended with 2 nodes, we need to join them
    if len(active_nodes) == 2:
        node_a, node_b = active_nodes
        # The distance between them is the remaining length
        # In NJ, the last two nodes are joined by a single edge? 
        # Actually, for unrooted tree, the last step joins the last two nodes.
        # The length is d(node_a, node_b).
        # But wait, NJ creates a binary tree. The last two nodes are connected by an edge.
        # Is there an internal node? No, the last two nodes are just connected.
        # However, our tree_edges structure needs to reflect this.
        # We just add an edge between them.
        len_ab = d[node_a, node_b]
        if len_ab < 0: len_ab = 0.0
        tree_edges[node_a].append((node_b, len_ab))
        tree_edges[node_b].append((node_a, len_ab))
        
    elif len(active_nodes) == 1:
        # This shouldn't happen for unrooted NJ with n>2, but just in case
        pass

    # Build Newick string
    # We need to traverse the tree.
    # Find a root (any node).
    # Since it's unrooted, we can pick any node and do a DFS.
    
    if not active_nodes and n_taxa > 0:
        # This case is handled if we broke early. 
        # If we broke early, we might have a fragmented tree.
        # For safety, if we didn't finish NJ, return a star tree or the best partial.
        # Given n<=32, NJ is very fast. This is a fallback.
        return "(" + ",".join(labels) + ");"
        
    # Find the root of the traversal
    # It's better to find a leaf or just pick the first node in the original list that exists in tree_edges
    # All nodes 0..n_taxa-1 are leaves.
    root = 0
    if root not in tree_edges:
         # Find any valid node
         for k in tree_edges:
             root = k
             break
             
    visited = set()
    
    def build_newick(node):
        if node in visited:
            return ""
        visited.add(node)
        
        if node_labels[node] is not None:
            return node_labels[node]
            
        children = []
        for neighbor, length in tree_edges[node]:
            if neighbor not in visited:
                child_str = build_newick(neighbor)
                if child_str:
                    children.append(f"{child_str}:{length:.4f}")
                    
        if not children:
            # This should not happen for internal nodes in a connected tree
            return f"n{node}"
            
        return "(" + ",".join(children) + ")"
        
    newick_str = build_newick(root)
    return newick_str + ";"
