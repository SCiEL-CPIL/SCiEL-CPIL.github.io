"""Phylogenetic topology recovery using optimized Neighbor Joining with NumPy."""
import numpy as np
import math
import time

def _jc_distance_matrix(seqs_array):
    """
    Compute the Jukes-Cantor corrected distance matrix from a NumPy array of sequence indices.
    seqs_array: shape (n_taxa, n_sites), values in {0, 1, 2, 3}
    """
    n_taxa, n_sites = seqs_array.shape
    if n_sites == 0:
        return np.zeros((n_taxa, n_taxa))
        
    # Compute pairwise Hamming distances
    # Using broadcasting: (n, 1, L) vs (1, n, L)
    diff = (seqs_array[:, np.newaxis, :] != seqs_array[np.newaxis, :, :])
    hamming = diff.sum(axis=2)
    
    p = hamming / n_sites
    
    # JC correction: d = -0.75 * ln(1 - 4/3 * p)
    # Clip p to avoid log(0) or log(negative)
    # Max p for log to be defined is 0.75.
    p_clipped = np.clip(p, 0, 0.7499)
    d = -0.75 * np.log(1.0 - (4.0 / 3.0) * p_clipped)
    
    # Set diagonal to 0
    np.fill_diagonal(d, 0.0)
    
    return d

def _neighbor_joining(dist_matrix, labels):
    """
    Perform Neighbor Joining on a distance matrix.
    Returns a Newick string.
    """
    n = len(labels)
    if n == 0:
        return ""
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    # Initialize distance dictionary
    dist = {}
    for i in range(n):
        for j in range(n):
            dist[(i, j)] = dist_matrix[i, j]
    
    active_nodes = list(range(n))
    next_node_id = n
    topology = {}
    node_labels = {i: labels[i] for i in range(n)}
    
    while len(active_nodes) > 2:
        # Compute row sums for active nodes
        r = {}
        for i in active_nodes:
            r[i] = sum(dist[(i, j)] for j in active_nodes if i != j)
        
        # Find min Q(i,j)
        min_q = float('inf')
        best_pair = None
        active_len = len(active_nodes)
        
        for idx_i in range(active_len):
            i = active_nodes[idx_i]
            for idx_j in range(idx_i + 1, active_len):
                j = active_nodes[idx_j]
                q = (active_len - 2) * dist[(i, j)] - r[i] - r[j]
                if q < min_q:
                    min_q = q
                    best_pair = (i, j)
        
        i, j = best_pair
        
        # Create new node k
        k = next_node_id
        next_node_id += 1
        
        # Update distances from k to all other active nodes m
        for m in active_nodes:
            if m == i or m == j:
                continue
            d_km = 0.5 * (dist[(i, m)] + dist[(j, m)] - dist[(i, j)])
            dist[(k, m)] = d_km
            dist[(m, k)] = d_km
        
        # Remove i and j from active, add k
        active_nodes.remove(i)
        active_nodes.remove(j)
        active_nodes.append(k)
        
        # Record topology
        topology[k] = (i, j)
        
    # Handle the last 2 nodes
    if len(active_nodes) == 2:
        u, v = active_nodes
        k = next_node_id
        topology[k] = (u, v)
        active_nodes.append(k)
    
    # Build Newick string
    def get_newick(node):
        if node in topology:
            left, right = topology[node]
            return f"({get_newick(left)},{get_newick(right)})"
        else:
            return node_labels[node]
    
    final_node = active_nodes[0]
    return get_newick(final_node) + ";"

def reconstruct(sequences):
    start_time = time.time()
    # 180s for 20 instances -> 9s per instance.
    # We will use 8s as a soft limit to be safe.
    deadline = start_time + 8.0
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return (labels[0] + ";") if n == 1 else ""
        
    # Map sequences to array
    seq_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seqs_array = np.array([[seq_map[c] for c in seq] for seq in (sequences[label] for label in labels)])
    
    # Compute distance matrix
    dist_matrix = _jc_distance_matrix(seqs_array)
    
    # Perform NJ
    newick = _neighbor_joining(dist_matrix, labels)
    
    return newick
