"""Phylogenetic topology recovery using Weighted UPGMA with vectorized distance calculation."""
# EVOLVE-BLOCK-START
import numpy as np

def reconstruct(sequences):
    """
    Reconstruct phylogenetic tree from sequences using Weighted UPGMA (WPGMA).
    This approach uses a hierarchical clustering method distinct from Neighbor Joining.
    Distances are computed using vectorized NumPy operations for speed.
    """
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n == 1:
        return labels[0]
    if n == 2:
        return f"({labels[0]},{labels[1]})"
        
    # 1. Compute JC Distance Matrix using NumPy
    # Convert sequences to a 2D numpy array of characters
    # For efficiency, we can compare directly
    seq_array = np.array([list(sequences[l]) for l in labels])
    L = seq_array.shape[1]
    
    # Calculate Hamming distances: (n, n)
    # Broadcasting: seq_array[:, None, :] != seq_array[None, :, :]
    # This creates an (n, n, L) boolean array
    hamming = np.sum(seq_array[:, None, :] != seq_array[None, :, :], axis=2)
    
    # Apply Jukes-Cantor correction
    p = hamming / L
    # Clip to avoid log(0) or invalid values
    p = np.clip(p, 0.0, 0.7499)
    jc_dist = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    
    # 2. Perform WPGMA
    # Data structures
    # active_nodes: list of current cluster/node IDs
    # node_children: dict node_id -> [child1, child2]
    # node_label: dict node_id -> label (for leaves)
    # current_dist: (n_max, n_max) distance matrix (we only use active parts)
    # current_weights: (n_max,) weights (number of taxa in cluster)
    
    max_nodes = 2 * n
    current_dist = np.zeros((max_nodes, max_nodes))
    current_dist[:n, :n] = jc_dist
    
    current_weights = np.zeros(max_nodes)
    current_weights[:n] = 1.0
    
    node_children = {}
    node_label = {}
    
    # Initialize leaves
    for i in range(n):
        node_children[i] = []
        node_label[i] = labels[i]
        
    active_nodes = list(range(n))
    next_id = n
    
    while len(active_nodes) > 1:
        # Find the pair with minimum distance
        min_d = float('inf')
        i_idx = -1
        j_idx = -1
        len_active = len(active_nodes)
        
        for i in range(len_active):
            for j in range(i + 1, len_active):
                ni = active_nodes[i]
                nj = active_nodes[j]
                d = current_dist[ni, nj]
                if d < min_d:
                    min_d = d
                    i_idx = i
                    j_idx = j
        
        ni = active_nodes[i_idx]
        nj = active_nodes[j_idx]
        
        # Create new internal node
        new_id = next_id
        next_id += 1
        
        node_children[new_id] = [ni, nj]
        node_label[new_id] = None
        
        # Update distances to other active nodes using WPGMA formula
        w_i = current_weights[ni]
        w_j = current_weights[nj]
        w_new = w_i + w_j
        
        # Vectorized update for distances to all other active nodes
        # We need to update current_dist[new_id, k] and current_dist[k, new_id]
        # for all k in active_nodes except ni, nj
        
        # Get indices of other active nodes
        other_indices = [idx for idx in active_nodes if idx not in (ni, nj)]
        
        if other_indices:
            # Fetch distances from ni and nj to others
            d_i = current_dist[ni, other_indices]
            d_j = current_dist[nj, other_indices]
            
            # WPGMA update
            d_new = (w_i * d_i + w_j * d_j) / w_new
            
            # Assign to new row/col
            current_dist[new_id, other_indices] = d_new
            current_dist[other_indices, new_id] = d_new
        
        # Update weights
        current_weights[new_id] = w_new
        
        # Update active nodes list
        active_nodes.remove(ni)
        active_nodes.remove(nj)
        active_nodes.append(new_id)
        
    # Root is the last remaining node
    root = active_nodes[0]
    
    # 3. Generate Newick String
    def get_newick(node_id):
        if not node_children[node_id]:
            return node_label[node_id]
        else:
            c1 = get_newick(node_children[node_id][0])
            c2 = get_newick(node_children[node_id][1])
            return f"({c1},{c2})"
            
    return get_newick(root)
# EVOLVE-BLOCK-END
